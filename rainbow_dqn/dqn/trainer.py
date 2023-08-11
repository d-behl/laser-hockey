import numpy as np
import time
from base.evaluator import evaluate
from laserhockey import hockey_env as h_env
from utils.utils import poll_opponent_dqn
from agent import DQNAgent
import copy

def dist_positions(p1, p2):
  return np.sqrt(np.sum(np.asarray(p1 - p2) ** 2, axis=-1))

MAX_PUCK_SPEED = 25
SCALE = 60.0

VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2
MAX_TIME_KEEP_PUCK = 15

def _get_info_2(env):
    # different proxy rewards:
    # Proxy reward/penalty for not being close to puck in the own half when puck is flying towards goal (not to opponent)
    reward_closeness_to_puck = 0
    if env.puck.position[0] > CENTER_X and env.puck.linearVelocity[0] >= 0:
      dist_to_puck = dist_positions(env.player2.position, env.puck.position)
      max_dist = 250. / SCALE
      max_reward = -30.  # max (negative) reward through this proxy
      factor = max_reward / (max_dist * env.max_timesteps / 2)
      reward_closeness_to_puck += dist_to_puck * factor  # Proxy reward for being close to puck in the own half
    # Proxy reward: touch puck
    reward_touch_puck = 0.
    if env.player2_has_puck == MAX_TIME_KEEP_PUCK:
      reward_touch_puck = 1.

    # puck is flying in the right direction
    max_reward = 1.
    factor = max_reward / (env.max_timesteps * MAX_PUCK_SPEED)
    reward_puck_direction = env.puck.linearVelocity[0] * factor  # Puck flies right is good and left not

    return {"winner": env.winner,
            "reward_closeness_to_puck": reward_closeness_to_puck,
            "reward_touch_puck": reward_touch_puck,
            "reward_puck_direction": reward_puck_direction,
            }

def calc_opposite_side_rew(env):
    r = 0

    if env.done:
      if env.winner == 0:  # tie
        r += 0
      elif env.winner == -1:  # you won
        r += 10
      else:  # opponent won
        # r -= 12.5
        r -= 10
    return r

class DQNTrainer:
    """
    The DQNTrainer class implements a trainer for the DQNAgent.

    Parameters
    ----------
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    config: dict
        The variable specifies config variables.
    """

    def __init__(self, logger, config) -> None:
        self.logger = logger
        self._config = config

    def train(self, agent, env, loaded_agents):
        epsilon = self._config['epsilon']
        epsilon_decay = self._config['epsilon_decay']
        min_epsilon = self._config['min_epsilon']
        episode_counter = 1
        total_step_counter = 0
        total_grad_updates = 0

        beta = self._config['per_beta']
        beta_inc = self._config['per_beta_inc']
        beta_max = self._config['per_beta_max']

        rew_stats = []
        loss_stats = []
        lost_stats = {}
        touch_stats = {}
        won_stats = {}

        eval_stats = {
            'reward': [],
            'touch': [],
            'won': [],
            'lost': []
        }

        curriculum = self._config['curriculum']
        repeats_remaining = 0
        self.current_mode = 'normal' # Just default
        last_reset_episode = 0
        trigger_reset_once = True
        playing_on_opposite_side = False

        if curriculum:
            # Messy as hell implementation, maybe one day in future I'll be more motivated
            # and change this. Right now, life is bleak.
            opponents = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)] #'shooting', 'defense',
            if len(loaded_agents)  != 0:
                opponents += loaded_agents
        else:
            opponents = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]

        while episode_counter <= self._config['max_episodes']:
            env.close() ## We will soon redefine environment

            # Here, if curriculum is set, mode is not considered at all
            # if it is False, then we select the environment according to mode
            if curriculum:
                if repeats_remaining > 0:
                    repeats_remaining -= 1
                else:
                    # if episode_counter < play_strong_after:
                    #     opponent = h_env.BasicOpponent(weak=True)
                    # else:
                    opponent, opponent_idx = poll_opponent_dqn(opponents=opponents)
                    if opponent == 'shooting' or opponent == 'defense':
                        repeats_remaining = self._config['num_repeats']
            elif self._config['mode'] == 'shooting':
                env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
                current_mode = 'shooting'
            elif self._config['mode'] == 'defense':
                env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
                current_mode = 'defense'               
            else:
                opponent = h_env.BasicOpponent(weak=False)
            
            ## If curriculum is selected we need to set environment and mode
            ## for things to work properly
            if curriculum:
                if opponent == 'shooting':
                    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
                    current_mode = 'shooting'
                elif opponent == 'defense':
                    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
                    current_mode = 'defense'
                else:
                    env = h_env.HockeyEnv()
                    current_mode = 'normal'

            other_side = np.random.uniform(0,1)
            if other_side < 0.5:
                playing_on_opposite_side = True
            else:
                playing_on_opposite_side = False

            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and current_mode == 'defense') or (
                    env.puck.position[0] > 5 and current_mode == 'shooting'):
                continue

            epsilon = max(epsilon - epsilon_decay, min_epsilon)
            if self._config['per']:
                beta = min(beta_max, beta + beta_inc)
                agent.update_per_beta(beta=beta)

            total_reward = 0
            touched = 0
            first_time_touch = 1
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0
            n_step_transition = []
            n_step = self._config['n_steps'] # No. of transitions for multistep

            for step in range(1, self._config['max_steps'] + 1):
                
                if playing_on_opposite_side:
                    a2 = agent.act(obs_agent2, eps=epsilon)
                    a2_list = agent.action_mapping[a2]

                    if current_mode == 'normal':
                        if opponent_idx > 1 and opponent_idx <= (1 + len(loaded_agents)):
                            a1 = opponent.act2(ob, eps=0)
                        elif opponent_idx <= 1:
                            a1 = opponent.act(ob)
                        else: ## Case when self play is done
                            a1 = opponent.act(ob, eps=0)
                        # a copy of our agent has been chosen, transform the action id to a list
                        if not isinstance(a1, np.ndarray):
                            a1 = agent.action_mapping[a1]
                    elif current_mode == 'shooting' or current_mode == 'defense':
                        a1 = [0, 0, 0, 0]
                    
                    (ob_new, reward, done, _, _info) = env.step(np.hstack([a1, a2_list]))
                    
                    # Used for next iteration update
                    ob_new_old = copy.deepcopy(ob_new)

                    ob_new = env.obs_agent_two()
                    _info = _get_info_2(env)
                    ob = obs_agent2
                    a1 = a2
                    touched = max(touched, _info['reward_touch_puck'])
                    reward = calc_opposite_side_rew(env) + _info['reward_closeness_to_puck']

                    step_reward = reward + 4 * _info['reward_closeness_to_puck'] - (1 - touched) * 0.1 + \
                                touched * first_time_touch * 0.1 * step

                    first_time_touch = 1 - touched

                    total_reward += step_reward
                    
                else:
                    a1 = agent.act(ob, eps=epsilon)
                    a1_list = agent.action_mapping[a1]

                    if current_mode == 'normal':
                        if opponent_idx > 1 and opponent_idx <= (1 + len(loaded_agents)):
                            a2 = opponent.act2(obs_agent2, eps=0)
                        elif opponent_idx <= 1:
                            a2 = opponent.act(obs_agent2)
                        else: ## Case when self play is done
                            a2 = opponent.act(obs_agent2, eps=0)
                        # a copy of our agent has been chosen, transform the action id to a list
                        if not isinstance(a2, np.ndarray):
                            a2 = agent.action_mapping[a2]
                    elif current_mode == 'shooting' or current_mode == 'defense':
                        a2 = [0, 0, 0, 0]
                    else:
                        raise NotImplementedError(f'Training for {current_mode} not implemented.')

                    (ob_new, reward, done, _, _info) = env.step(np.hstack([a1_list, a2]))

                    touched = max(touched, _info['reward_touch_puck'])

                    # if _info['winner'] == -1:
                    #     reward -= 2.5 # Now for losing we have -12.5 reward
                    # elif _info['winner'] == 1:
                    #     reward += 15
                    step_reward = reward + 4 * _info['reward_closeness_to_puck'] - (1 - touched) * 0.1 + \
                                touched * first_time_touch * 0.1 * step

                    first_time_touch = 1 - touched

                    total_reward += step_reward

                if self._config['multi_step']:
                    if( step < n_step):
                        n_step_transition.append((ob, a1, step_reward, ob_new, done))
                    else:
                        n_step_transition.append((ob, a1, step_reward, ob_new, done))
                        agent.store_transition(n_step_transition)
                        n_step_transition.pop(0)
                else:
                    agent.store_transition((ob, a1, step_reward, ob_new, done))

                if self._config['show']:
                    time.sleep(0.01)
                    env.render()

                if touched > 0:
                    touch_stats[episode_counter] = 1

                if done:
                    if playing_on_opposite_side:
                        won_stats[episode_counter] = 1 if env.winner == -1 else 0
                        lost_stats[episode_counter] = 1 if env.winner == 1 else 0
                    else:
                        won_stats[episode_counter] = 1 if env.winner == 1 else 0
                        lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break
                
                if total_step_counter % self._config['train_every'] == 0 and \
                        total_step_counter > self._config['start_learning_from']:

                    loss_stats.append(agent.train_model())
                    rew_stats.append(total_reward)
                    total_grad_updates += 1

                    if total_grad_updates % self._config['update_target_every'] == 0:
                        agent.update_target_net()

                    if self._config['self_play'] and total_grad_updates % self._config['add_opponent_every'] == 0 and \
                            episode_counter >= self._config['start_self_play_from']:
                        ## If reset is set, we will reset once new player is added
                        if self._config['reset_branch'] or self._config['reset_last_layer']:
                            if episode_counter - last_reset_episode > self._config['reset_wait_eps']:
                                opponents.append(copy_agent(agent))
                                agent.id += 1
                                if self._config['reset_branch'] and trigger_reset_once:
                                    agent.reset_last_branch()
                                    ## We need to reset epsilon too
                                    epsilon = self._config['epsilon']
                                elif self._config['reset_last_layer'] and trigger_reset_once:
                                    agent.reset_last_layer()
                                    ## We need to reset epsilon too
                                    epsilon = self._config['epsilon']
                                trigger_reset_once = False
                        else:
                            opponents.append(copy_agent(agent))
                            agent.id += 1

                if playing_on_opposite_side:
                    ob = ob_new_old
                    obs_agent2 = env.obs_agent_two()
                else:
                    ob = ob_new
                    obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

            # self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon, touched, opponent)
            if episode_counter % 200 == 0:
                print("Current Episode: ", episode_counter)
            if episode_counter % self._config['evaluate_every'] == 0:
                self.logger.info("Evaluating agent")
                agent.eval()
                old_show = agent._config['show']
                agent._config['show'] = False
                if not curriculum:
                    current_mode = self._config['mode']
                    rew, touch, won, lost = evaluate(agent=agent, env=env,current_mode=current_mode ,
                                                     opponent=h_env.BasicOpponent(weak=False),
                                                    eval_episodes=self._config['eval_episodes'], quiet=True,
                                                    action_mapping=agent.action_mapping)
                else:
                    env.close()
                    # eval_episodes = 200
                    # self.logger.info("Evaluating SHOOTING")
                    # current_mode='shooting'
                    # env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
                    # _, _, _, _ = evaluate(agent=agent, env=env,current_mode=current_mode ,
                    #                                  opponent='shooting',
                    #                                 eval_episodes=eval_episodes, quiet=True,
                    #                                 action_mapping=agent.action_mapping)
                    # env.close()

                    # self.logger.info("Evaluating DEFENSE")
                    # current_mode='defense'
                    # env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
                    # _, _, _, _ = evaluate(agent=agent, env=env,current_mode=current_mode ,
                    #                                  opponent='defense',
                    #                                 eval_episodes=eval_episodes, quiet=True,
                    #                                 action_mapping=agent.action_mapping)
                    # env.close()

                    self.logger.info("Evaluating WEAK")
                    current_mode='normal'
                    env = h_env.HockeyEnv()
                    rew, touch, won, lost = evaluate(agent=agent, env=env,current_mode=current_mode ,
                                                     opponent=h_env.BasicOpponent(weak=True),
                                                    eval_episodes=self._config['eval_episodes'], quiet=True,
                                                    action_mapping=agent.action_mapping)
                    env.close()

                    self.logger.info("Evaluating STRONG")
                    current_mode='normal'
                    env = h_env.HockeyEnv()
                    rew, touch, won, lost = evaluate(agent=agent, env=env,current_mode=current_mode ,
                                                     opponent=h_env.BasicOpponent(weak=False),
                                                    eval_episodes=self._config['eval_episodes'], quiet=True,
                                                    action_mapping=agent.action_mapping)

                    eval_stats['reward'].append(rew)
                    eval_stats['touch'].append(touch)
                    eval_stats['won'].append(won)
                    eval_stats['lost'].append(lost)

                    self.logger.info("Evaluating OPPOSITE")
                    current_mode='normal'
                    rew, touch, won, lost = evaluate(agent=agent, env=env,current_mode=current_mode ,
                                                     opponent=h_env.BasicOpponent(weak=False),
                                                    eval_episodes=self._config['eval_episodes'], quiet=True,
                                                    action_mapping=agent.action_mapping,
                                                    evaluate_on_opposite_side=True)
                    

                agent.train()
                agent._config['show'] = old_show

                eval_stats['reward'].append(rew)
                eval_stats['touch'].append(touch)
                eval_stats['won'].append(won)
                eval_stats['lost'].append(lost)
                save_file_name = self._config['save_prefix'] +"_"+str(episode_counter) + '.pkl'
                self.logger.save_model(agent, save_file_name)

            if total_step_counter > self._config['start_learning_from']:
                agent.step_lr_scheduler()

            episode_counter += 1

        if self._config['show']:
            env.close()

        # Print train stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        self.logger.info('Saving statistics...')

        # Plot reward
        self.logger.plot_running_mean(rew_stats, 'Total reward', self._config['save_prefix'] + 'total-reward.pdf', show=False)

        # Plot loss
        self.logger.plot_running_mean(loss_stats, 'Loss', self._config['save_prefix'] + 'loss.pdf', show=False)

        # Plot evaluation stats
        self.logger.plot_intermediate_stats(eval_stats, show=False)

        # Save model
        filename = self._config['save_prefix'] + '_final_model.pkl'
        self.logger.save_model(agent, filename)

        # Save arrays of won-lost stats
        self.logger.save_array(data=eval_stats["won"], filename=self._config['save_prefix'] + "eval-won-stats")
        self.logger.save_array(data=eval_stats["lost"], filename=self._config['save_prefix'] + "eval-lost-stats")

def copy_agent(agent):
    #Added to avoid deepcopy failure
    newAgent = DQNAgent(
        logger=agent.logger,
        obs_dim=agent.obs_dim,
        action_mapping=agent.action_mapping,
        userconfig=agent._config
    )
    newAgent.id = agent.id
    newAgent.Q.load_state_dict(agent.Q.state_dict())
    newAgent.update_target_net()
    return newAgent
