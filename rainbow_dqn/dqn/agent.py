import sys
sys.path.insert(0, '..')

import copy
import numpy as np
from qfeedforward import QFunction
from base.agent import Agent
import torch.nn as nn


class DQNAgent(Agent):
    """
    The DQNAgent class implements a trainable DQN agent.

    Parameters
    ----------
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_mapping: Iterable
        The variable specifies a custom action space
    **userconfig:
        The variable specifies the config settings.
    """

    def __init__(self, logger, obs_dim, action_mapping, userconfig):
        super().__init__(
            logger=logger,
            obs_dim=obs_dim,
            action_dim=len(action_mapping),
            userconfig=userconfig
        )
        self.id = 1
        self.action_mapping = action_mapping

        lr_milestones = []
        if self._config['lr_milestones'] is not None:
            lr_milestones = [int(x) for x in (self._config['lr_milestones']).split(' ')]
        else:
            lr_milestones = np.arange(start=0,
                                      stop=self._config['max_episodes'] + 1,
                                      step=self._config['change_lr_every'])[1:]

        self.Q = QFunction(
            obs_dim=obs_dim,
            action_dim=len(action_mapping),
            hidden_sizes=self._config['hidden_sizes'],
            learning_rate=self._config['learning_rate'],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device'],
            dueling=self._config['dueling'],
            proximal_loss =self._config['proximal_loss'],
            spectral=self._config['spectral_norm'],
            weight_norm=self._config['weight_norm'],
            noisyNets = self._config['NoisyNets']
        )

        # self.target_Q = copy.deepcopy(self.Q)
        self.target_Q = QFunction(
                obs_dim=obs_dim,
                action_dim=len(action_mapping),
                hidden_sizes=self._config['hidden_sizes'],
                learning_rate=self._config['learning_rate'],
                lr_milestones=lr_milestones,
                lr_factor=self._config['lr_factor'],
                device=self._config['device'],
                dueling=self._config['dueling'],
                proximal_loss =self._config['proximal_loss'],
                spectral=self._config['spectral_norm'],
                weight_norm=self._config['weight_norm'],
                noisyNets = self._config['NoisyNets']
            )
        
        self.Q.to(self._config['device'])
        self.target_Q.to(self._config['device'])
        self.update_target_net()

    def train(self):
        self.Q.train()
        self.target_Q.train()

    def eval(self):
        self.Q.eval()
        self.target_Q.eval()

    def update_target_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    # Reset last layers that is pre + final layer
    def reset_last_branch(self):
        ## Reset the learning rate for specific layers
        for param_group in self.Q.optimizer.param_groups:
            # print(param_group['name'])
            if self._config['dueling']:
                if param_group['name'] == 'pre_A' or \
                    param_group['name'] == 'A' or\
                    param_group['name'] == 'pre_V' or\
                    param_group['name'] == 'V':

                    param_group['lr'] = 0.00005
            else:
                if param_group['name'] == 'pre_Q' or \
                    param_group['name'] == 'Q':

                    param_group['lr'] = 0.00005
            
        ## Currently NoisyNets are only enabled with Dueling.
        ## TO DO: Add noisy with normal too
        if self._config['NoisyNets']:
            self.Q.pre_A.parameter_initialization()
            self.Q.A.parameter_initialization()
            self.Q.pre_V.parameter_initialization()
            self.Q.V.parameter_initialization()
        else:
            for name, param in self.Q.named_parameters():
                if self._config['dueling']:
                    if 'pre_A' in name or 'A' in name \
                        or 'pre_V' in name or 'V' in name:
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                nn.init.zeros_(param)
                else:
                    if 'pre_Q' in name or 'Q' in name:
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.zeros_(param)
        self.update_target_net()

    ## Just reset very last layer  
    def reset_last_layer(self):
        ## Reset the learning rate for specific layers
        for param_group in self.Q.optimizer.param_groups:
            # print(param_group['name'])
            if self._config['dueling']:
                    
                if  param_group['name'] == 'A' or\
                    param_group['name'] == 'V':

                    param_group['lr'] = 0.00005
            else:
                if param_group['name'] == 'Q':

                    param_group['lr'] = 0.00005
            
        ## Currently NoisyNets are only enabled with Dueling.
        ## TO DO: Add noisy with normal too
        if self._config['NoisyNets']:
            self.Q.A.parameter_initialization()
            self.Q.V.parameter_initialization()
        else:
            for name, param in self.Q.named_parameters():
                if self._config['dueling']:
                    if 'A' in name or 'V' in name:
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                nn.init.zeros_(param)
                else:
                    if 'Q' in name:
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.zeros_(param)
        self.update_target_net()

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._config['epsilon']
        
        if self._config['NoisyNets']:
            action = self.Q.greedyAction(observation)
        
        else:
            if np.random.random() > eps:
                action = self.Q.greedyAction(observation)
            else:
                action = np.random.randint(0, len(self.action_mapping))

        return action
    
    def act2(self, observation, eps=None):
        if eps is None:
            eps = self._config['epsilon']
        
        if self._config['NoisyNets']:
            action = self.Q.greedyAction2(observation)
        
        else:
            if np.random.random() > eps:
                action = self.Q.greedyAction2(observation)
            else:
                action = np.random.randint(0, len(self.action_mapping))

        return action

    def enable_proximal_loss(self):
        self.Q.proximal_loss = True

    def train_model(self):
        if self._config['multi_step']:
            data_multi = self.buffer.sample(batch_size=self._config['batch_size'], multi_step=True)
            data_last = data_multi[0][:,-1]
            data_first = data_multi[0][:,0]

            s = np.stack(data_first[:, 0])
            a = np.stack(data_first[:,1])[:, None]
            s_next = np.stack(data_last[:, 3])
            rew = np.zeros((self._config['batch_size'],1), dtype = float)
            discount = 1
            for i in range(self._config['n_steps']):
                cur_rew = data_multi[0][:,i, 2][:, None].astype(float)
                rew = rew + discount * cur_rew
                discount = discount * self._config['discount']

            not_done = (~np.stack(data_last[:, 4])[:, None]).astype(int)  # not_done flag
        else:
            data = self.buffer.sample(batch_size=self._config['batch_size'])
            discount = self._config['discount']
            s = np.stack(data[:, 0])  # s_t
            a = np.stack(data[:, 1])[:, None]  # a_t
            rew = np.stack(data[:, 2])[:, None]  # r
            s_next = np.stack(data[:, 3])  # s_t+1
            not_done = (~np.stack(data[:, 4])[:, None]).astype(int)  # not_done flag

        if self._config['double']:
            greedy_actions = self.Q.greedyAction(s_next)[:, None]
            value_s_next = self.target_Q.Q_value(s_next, greedy_actions).detach().cpu().numpy()
        else:
            value_s_next = self.target_Q.maxQ(s_next)[:, None]

        targets = rew + discount * np.multiply(not_done, value_s_next)

        if self._config['per']:
            if self._config['multi_step']:
                weights = np.stack(data_multi[1])
                indices = np.stack(data_multi[2].flatten())
            else:
                weights = np.stack(data[:, 5])[:, None]
                indices = np.stack(data[:, 6])
            # test = data_full[1]
            
        else:
            weights = np.ones(targets.shape)

        # optimize
        fit_loss, pred = self.Q.fit(s, a, targets, weights, self.target_Q)

        if self._config['per']:
            priorities = np.abs(targets - pred) + 1e-6
            self.buffer.update_priorities(indices=indices, priorities=priorities.flatten())

        return fit_loss

    def update_per_beta(self, beta):
        self.buffer.update_beta(beta=beta)

    def step_lr_scheduler(self):
        self.Q.lr_scheduler.step()

    def __str__(self):
        return f"DQN {self.id}"
