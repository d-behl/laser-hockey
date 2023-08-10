import numpy as np
import time

from base.evaluator import evaluate
from sac_agent import SACAgent
from utils import *
from laserhockey import hockey_env as h_env


from torch.utils.tensorboard import SummaryWriter

#from pink import PinkNoiseDist

class SACTrainer:
    """
    The SACTrainer class implements a trainer for the SACAgent.

    Parameters
    ----------
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    args: dict
        The variable specifies args variables.
    """

    def __init__(self, logger, args) -> None:
        self.logger = logger
        self.args = args
        self.writer = SummaryWriter(logger.prefix_path)

    def plot_status(self, rew_stats, eval_stats, q1_losses, q2_losses, actor_losses, alpha_losses, new_op_grad):
        # Plot reward
        self.logger.plot_running_mean(data=rew_stats, title='Total reward', filename='total-reward.pdf', show=False)

        # Plot evaluation stats
        self.logger.plot_evaluation_stats(eval_stats, self.args.evaluate_every, 'evaluation-won-lost.pdf')

        # Plot losses
        for loss, title in zip([q1_losses, q2_losses, actor_losses, alpha_losses],
                                ['Q1 loss', 'Q2 loss', 'Policy loss', 'Alpha loss']):
            self.logger.plot_running_mean(
                data=loss,
                title=title,
                filename=f'{title.replace(" ", "-")}.pdf',
                show=False,
                v_milestones=new_op_grad,
            )
            

    def train(self, agent, opponents, env):
        rew_stats, q1_losses, q2_losses, actor_losses, alpha_losses = [], [], [], [], []

        # For pinknoise
        #seq_len = 1000
        #action_dim = env.num_actions
        #noise = PinkNoiseDist(seq_len, action_dim)

        lost_stats, touch_stats, won_stats = {}, {}, {}
        eval_stats = {
            'weak': {
                'reward': [],
                'touch': [],
                'won': [],
                'lost': []
            },
            'strong': {
                'reward': [],
                'touch': [],
                'won': [],
                'lost': []
            }
        }

        episode_counter = 1
        total_step_counter = 0
        grad_updates = 0
        new_op_grad = []
        while episode_counter <= self.args.max_episodes:
            total_reward, touched = 0, 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0
            first_time_touch = 1
            if self.args.opposite_side:
                is_opposite = np.random.choice([True, False], p=[0.5, 0.5])
            if self.args.opposite_side and is_opposite:
                ob, info1 = env.reset()
                obs_agent2 = env.obs_agent_two()
                info2 = env.get_info_agent_two()
                phase = calculate_phase(obs=obs_agent2, env=env, info=info2, player=2)
                opponent = poll_opponent(opponents)
                for step in range(self.args.max_steps):
                    if self.args.phased:
                        a2 = agent.act(obs_agent2, phase=phase)
                    else:
                        a2 = agent.act(obs_agent2)
                    # if self.args.noise: 
                    #     a2 += noise.sample()

                    if self.args.mode == 'defense':
                        a1 = opponent.act(ob)
                    elif self.args.mode == 'shooting':
                        a1 = np.zeros_like(a2)
                    else:
                        a1 = opponent.act(ob)

                    actions = np.hstack([a1, a2])
                    next_state, _, d, t, info1 = env.step(actions)
                    done = d
                    info2 = env.get_info_agent_two()
                    reward = env.get_reward_agent_two(info2)
                    next_state2 = env.obs_agent_two()
                    if self.args.phased:
                        next_phase = calculate_phase(obs=next_state2, env=env, info=info2, player=2)
                    
                    touched = max(touched, info2['reward_touch_puck'])
                    step_reward = (
                        reward
                        + info2['reward_puck_direction']
                        + info2['reward_closeness_to_puck']
                        - (1 - touched) * 0.1
                        + touched * first_time_touch * 0.1
                    )
                    first_time_touch = 1 - touched

                    total_reward += step_reward
                    if self.args.phased:
                        agent.store_transition((obs_agent2, a2, step_reward, next_state2, done, phase, next_phase))
                    else:
                        agent.store_transition((obs_agent2, a2, step_reward, next_state2, done))

                    if self.args.show:
                        env.render()

                    if touched > 0:
                        touch_stats[episode_counter] = 1

                    if d or t:
                        lost_stats[episode_counter] = 1 if env.winner == 1 else 0
                        won_stats[episode_counter] = 1 if env.winner == -1 else 0
                        break

                    
                    ob = next_state
                    obs_agent2 = next_state2
                    phase = next_phase
                    total_step_counter += 1 
            else:
                ob, info1 = env.reset()
                obs_agent2 = env.obs_agent_two()
                assert ob[12] == -obs_agent2[12]

                opponent = poll_opponent(opponents)
                phase = calculate_phase(obs=ob, env=env, info=info1, player=1)
                for step in range(self.args.max_steps):
                    self.writer.add_scalar('phase', phase, total_step_counter)
                    if self.args.phased:
                        a1 = agent.act(ob, phase=phase)
                    else:
                        a1 = agent.act(ob)
                    # if self.args.noise: 
                    #     a1 += noise.sample()

                    if self.args.mode == 'defense':
                        a2 = opponent.act(obs_agent2)
                    elif self.args.mode == 'shooting':
                        a2 = np.zeros_like(a1)
                    else:
                        a2 = opponent.act(obs_agent2)


                    actions = np.hstack([a1, a2])
                    next_state, reward, d, t, info1 = env.step(actions)
                    if self.args.phased:
                        next_phase = calculate_phase(obs=next_state, env=env, info=info1, player=1)
                    done = d
                    
                    touched = max(touched, info1['reward_touch_puck'])

                    step_reward = (
                        reward
                        + info1['reward_puck_direction']
                        + 5 * info1['reward_closeness_to_puck']
                        - (1 - touched) * 0.1
                        + touched * first_time_touch * 0.1
                    )
                    first_time_touch = 1 - touched

                    total_reward += step_reward
                    if self.args.phased:
                        agent.store_transition((ob, a1, step_reward, next_state, done, phase, next_phase))
                    else:
                        agent.store_transition((ob, a1, step_reward, next_state, done))

                    if self.args.show:
                        env.render()

                    if touched > 0:
                        touch_stats[episode_counter] = 1

                    if d or t:
                        won_stats[episode_counter] = 1 if env.winner == 1 else 0
                        lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                        break


                    ob = next_state
                    obs_agent2 = env.obs_agent_two()
                    phase = next_phase
                    total_step_counter += 1

            if agent.buffer.size < self.args.batch_size:
                continue

            for _ in range(self.args.grad_steps):
                losses = agent.update_parameters(total_step_counter)
                grad_updates += 1

                q1_losses.append(losses[0])
                q2_losses.append(losses[1])
                actor_losses.append(losses[2])
                alpha_losses.append(losses[3])

                # Add trained agent to opponents queue
                if self.args.selfplay:
                    if (
                        grad_updates % self.args.add_self_every == 0
                    ):
                        new_opponent = SACAgent.clone_from(agent)
                        new_opponent.eval()
                        opponents.append(new_opponent)
                        new_op_grad.append(grad_updates)

            agent.schedulers_step()
            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward)

            if episode_counter % self.args.evaluate_every == 0:
                agent.eval()
                for eval_op in ['strong', 'weak']:
                    ev_opponent = opponents[0] if eval_op == 'strong' else h_env.BasicOpponent(False)
                    rew, touch, won, lost = evaluate(
                        agent,
                        env,
                        ev_opponent,
                        100,
                        quiet=True
                    )
                    eval_stats[eval_op]['reward'].append(rew)
                    eval_stats[eval_op]['touch'].append(touch)
                    eval_stats[eval_op]['won'].append(won)
                    eval_stats[eval_op]['lost'].append(lost)
                agent.train()

                self.logger.save_model(agent, f'a-{episode_counter}.pkl')

            rew_stats.append(total_reward)
            self.writer.add_scalar('Total Reward', total_reward, episode_counter)
            self.writer.add_scalar('Q1 Loss', q1_losses[-1], episode_counter)
            self.writer.add_scalar('Q2 Loss', q2_losses[-1], episode_counter)
            self.writer.add_scalar('Actor Loss', actor_losses[-1], episode_counter)
            self.writer.add_scalar('Alpha Loss', alpha_losses[-1], episode_counter)

            episode_counter += 1

        if self.args.show:
            env.close()

        # Print train stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        self.logger.info('Saving training statistics...')

        #Use plot_status to plot the training statistics
        self.plot_status(rew_stats, eval_stats, q1_losses, q2_losses, actor_losses, alpha_losses, new_op_grad)

        # Save agent
        self.save_agent(agent)
        
        if self.args.evaluate:
            agent.eval()
            agent.args.show = True
            evaluate(agent, env, h_env.BasicOpponent(weak=False), self.args.eval_episodes)

    def save_agent(self, agent):   
        self.logger.save_model(agent, 'agent.pkl')
        self.logger.save_args(self.args)
        agent.save_model(self.logger.prefix_path)

    