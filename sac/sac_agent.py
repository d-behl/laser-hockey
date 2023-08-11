# The skeleton of the code here is adapted from https://github.com/anticdimi/laser-hockey.git
# Updates are changed to include phased setting.

import sys

import copy
sys.path.insert(0, '.')
sys.path.insert(1, '..')
import numpy as np
from pathlib import Path
import pickle

from base.agent import Agent
from models import ActorNetwork, CriticNetwork
from pmlp_actor import PMLPActor
from pmlp_critic import PMLPCritic
from utils import hard_update, soft_update
from base.experience_replay import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACAgent():
    """
    The SACAgent class implements a trainable Soft Actor Critic agent, as described in: https://arxiv.org/pdf/1812.05905.pdf.

    Parameters
    ----------
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_space: ndarray
        The variable specifies the action space of environment.
    args:
        The variable specifies the args settings.
    """

    def __init__(self, logger, obs_dim, action_space, args, action_dim):
        self.logger = logger

        if args.mode not in ['normal', 'shooting', 'defense']:
            raise ValueError('Unknown training mode. See --help')
        

        self.obs_dim = obs_dim
        self.args = args
        self.args.buffer_size = int(1e5)
        self.args.per_beta = 0.4

        if self.args.per:
            self.buffer = PrioritizedExperienceReplay(max_size=self.args.buffer_size,
                                                      alpha=self.args.per_alpha,
                                                      beta=self.args.per_beta)
        else:
            self.buffer = UniformExperienceReplay(max_size=int(1e5))

        self.action_space = action_space
        self.device = args.device
        self.alpha = args.alpha
        self.automatic_entropy_tuning = self.args.automatic_entropy_tuning
        self.eval_mode = False

        if self.args.lr_milestones is None:
            raise ValueError('lr_milestones argument cannot be None!\nExample: --lr_milestones=100 200 300')

        lr_milestones = [int(x) for x in (self.args.lr_milestones[0]).split(' ')]

        self.action_dim = action_dim

        if args.phased:
            self.actor = PMLPActor(
                input_size=obs_dim[0],
                hidden_size=[256,256],
                output_size=64,
                dtype=torch.float,
                n_layers=2,
                batch_size=self.args.batch_size,
                action_space=self.action_space,
                scale=1.0,
                action_dim = self.action_dim,
                args=args
            )
        else:
            self.actor = ActorNetwork(
                input_dims=obs_dim,
                learning_rate=self.args.learning_rate,
                action_space=self.action_space,
                hidden_sizes=[256, 256],
                lr_milestones=lr_milestones,
                lr_factor=self.args.lr_factor,
                action_dim = self.action_dim,
                device=self.args.device
            ).to(self.device)

        if args.phased:
            self.critic = PMLPCritic(
                input_size_state=obs_dim[0],
                input_size_action=self.action_dim,
                output_size=64,
                learning_rate=self.args.learning_rate,
                device=self.args.device,
                lr_milestones=lr_milestones,
                hidden_sizes=[256, 256],
                dtype=torch.float,
                n_layers=2,
                batch_size=self.args.batch_size,
                scale=1.0,
                lr_factor=self.args.lr_factor,
                loss='l2',
            )

            self.critic_target = PMLPCritic(
                input_size_state=obs_dim[0],
                input_size_action=self.action_dim,
                output_size=64,
                learning_rate=self.args.learning_rate,
                device=self.args.device,
                lr_milestones=lr_milestones,
                hidden_sizes=[256, 256],
                dtype=torch.float,
                n_layers=2,
                batch_size=self.args.batch_size,
                scale=1.0,
                lr_factor=self.args.lr_factor,
                loss='l2',
            )
        else:
            self.critic = CriticNetwork(
                input_dim=obs_dim,
                n_actions=self.action_dim,
                learning_rate=self.args.learning_rate,
                hidden_sizes=[256, 256],
                lr_milestones=lr_milestones,
                lr_factor=self.args.lr_factor,
                device=self.args.device
            ).to(self.device)

            self.critic_target = CriticNetwork(
                input_dim=obs_dim,
                n_actions=self.action_dim,
                learning_rate=self.args.learning_rate,
                hidden_sizes=[256, 256],
                lr_milestones=lr_milestones,
                device=self.args.device
            ).to(self.device)

        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning:
            milestones = [int(x) for x in (self.args.alpha_milestones[0]).split(' ')]
            self.target_entropy = -torch.tensor(4).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
            self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.alpha_optim, milestones=milestones, gamma=0.5
            )

    @classmethod
    def clone_from(cls, agent):
        clone = cls(
            copy.deepcopy(agent.logger),
            copy.deepcopy(agent.obs_dim),
            copy.deepcopy(agent.action_space),
            copy.deepcopy(agent.args),
            copy.deepcopy(agent.action_dim)
        )
        clone.critic.load_state_dict(agent.critic.state_dict())
        clone.critic_target.load_state_dict(agent.critic_target.state_dict())
        clone.actor.load_state_dict(agent.actor.state_dict())

        return clone
    
    @staticmethod
    def load_model_old(fpath):
        with open(Path(fpath), 'rb') as inp:
            return pickle.load(inp)

    def load_model(self, fpath, keep_args=False):
        agent = SACAgent.load_model_old(fpath)
        self.actor = copy.deepcopy(agent.actor)
        self.critic = copy.deepcopy(agent.critic)
        self.critic_target = copy.deepcopy(agent.critic_target)
        self.buffer =  copy.deepcopy(agent.buffer)
        if keep_args:
            self.args = agent.args

    def save_model(self, fpath):
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor, Path(fpath) / 'actor.pt')
        torch.save(self.critic, Path(fpath) / 'critic.pt')
        torch.save(self.critic_target, Path(fpath) / 'critic_target.pt')
        
    def store_transition(self, transition):
        if len(transition)>6:
            NotImplementedError('Does not accept extra data dimensions')
        self.buffer.add_transition(transition)

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def act(self, obs, phase=None):
        if self.args.phased:
            return self._act(obs, phase=phase, evaluate=True) if self.eval_mode else self._act(obs, phase=phase)
        return self._act(obs, evaluate=True) if self.eval_mode else self._act(obs)

    def _act(self, obs, evaluate=False, phase=None):
        # if isinstance(obs, tuple):
        #     obs = obs[0]
        state = torch.Tensor(obs).to(self.args.device).unsqueeze(0)
        if evaluate is False:
            if self.args.phased:
                action, _, _, _ = self.actor.sample(state, phase=phase)
            else:
                action, _, _, _ = self.actor.sample(state)
        else:
            if self.args.phased:
                _, _, action, _ = self.actor.sample(state, phase=phase)
            else:
                _, _, action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.actor.lr_scheduler.step()

    def update_parameters(self, total_step):
        data = self.buffer.sample(self.args.batch_size)
        if self.args.phased:
            if data.shape[1] > 6:
                phase = torch.tensor( 
                    np.stack(data[:, 5]),
                    device=self.device,
                    dtype=torch.float
                )
                next_phase = torch.tensor(
                    np.stack(data[:, 6]),
                    device=self.device,
                    dtype=torch.float
                )
        
        state = torch.tensor(
            np.stack(data[:, 0]),
            device=self.device,
            dtype=torch.float
        )

        next_state = torch.tensor(
            np.stack(data[:, 3]),
            device=self.device,
            dtype=torch.float
        )

        action = torch.tensor(
            np.stack(data[:, 1])[:, None],
            device=self.device,
            dtype=torch.float
        ).squeeze(dim=1)

        reward = torch.tensor(
            np.stack(data[:, 2])[:, None],
            device=self.device,
            dtype=torch.float
        ).squeeze(dim=1)

        not_done = torch.tensor(
            (~np.stack(data[:, 4])[:, None]).astype(int),
            device=self.device,
            dtype=torch.float
        ).squeeze(dim=1)

        with torch.no_grad():
            if self.args.phased:
                next_state_action, next_state_log_pi, _, _ = self.actor.sample(next_state, phase=next_phase)
                q1_next_targ, q2_next_targ = self.critic_target(next_state, next_state_action, phase=next_phase)
            else:
                next_state_action, next_state_log_pi, _, _ = self.actor.sample(next_state)
                q1_next_targ, q2_next_targ = self.critic_target(next_state, next_state_action)
            
            min_qf_next_target = torch.min(q1_next_targ, q2_next_targ) - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self.args.gamma * (min_qf_next_target).squeeze()
        if self.args.phased:
            qf1, qf2 = self.critic(state, action, phase=phase)
        else:
            qf1, qf2 = self.critic(state, action)

        qf1_loss = self.critic.loss(qf1.squeeze(), next_q_value)
        qf2_loss = self.critic.loss(qf2.squeeze(), next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        qf_loss.backward()
        self.critic.optimizer.step()

        if self.args.phased:
            pi, log_pi, _, _ = self.actor.sample(state, phase=phase)
        else:
            pi, log_pi, _, _ = self.actor.sample(state)
        if self.args.phased:
            qf1_pi, qf2_pi = self.critic(state, pi, phase=phase)
        else:
            qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean(axis=0)

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_scheduler.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if total_step % self.args.update_target_every == 0:
            soft_update(self.critic_target, self.critic, self.args.soft_tau)
        if self.args.phased:
            self.critic.update_control_gradients()
            self.actor.update_control_gradients()
            

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item())
    


