import time
import numpy as np


def evaluate(agent, env, opponent, eval_episodes, quiet=False, action_mapping=None, evaluate_on_opposite_side=False):
    old_verbose = env.verbose
    env.verbose = not quiet

    rew_stats = []
    touch_stats = {}
    won_stats = {}
    lost_stats = {}

    for episode_counter in range(eval_episodes):
        total_reward = 0
        ob, info_dict  = env.reset()
        obs_agent2 = env.obs_agent_two()

        if (env.puck.position[0] < 5 and agent.args.mode == 'defense') or (
                env.puck.position[0] > 5 and agent.args.mode == 'shooting'
        ):
            continue

        touch_stats[episode_counter] = 0
        won_stats[episode_counter] = 0
        lost_stats[episode_counter] = 0
        for step in range(env.max_timesteps):

            if evaluate_on_opposite_side:
                if action_mapping is not None:
                    if agent.args.phased:
                        a2 = agent.act(obs_agent2, phase=[info_dict['reward_puck_direction']], eps=0)
                    else:
                        a2 = agent.act(obs_agent2, eps=0) #TODO: fix this
                    a2 = action_mapping[a2]
                else:
                    if agent.args.phased:
                        a2 = agent.act(ob, phase=[info_dict['reward_puck_direction']])
                    else:
                        a2 = agent.act(ob)

                if agent.args.mode in ['defense', 'normal']:
                    a1 = opponent.act(ob)
                    if not isinstance(a1, np.ndarray):
                        a1 = action_mapping[a1]
                elif agent.args.mode == 'shooting':
                    a1 = [0, 0, 0, 0]
                else:
                    a1 = opponent.act(ob)

            else:
                if action_mapping is not None:
                    # DQN act
                    a1 = agent.act(ob, eps=0)
                    a1 = action_mapping[a1]
                else:
                    # SAC act
                    if agent.args.phased:
                        a1 = agent.act(ob, phase=[info_dict['reward_puck_direction']])
                    else:
                        a1 = agent.act(ob)

                if agent.args.mode in ['defense', 'normal']:
                    a2 = opponent.act(obs_agent2)
                    if not isinstance(a2, np.ndarray):
                        a2 = action_mapping[a2]
                elif agent.args.mode == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {agent.args.mode} not implemented.')

            (ob_new, reward, done, _, _info) = env.step(np.hstack([a1, a2]))
            ob = ob_new
            obs_agent2 = env.obs_agent_two()

            if evaluate_on_opposite_side:
                # Not really a way to implement this, given the structure of the env...
                touch_stats[episode_counter] = 0
                total_reward -= reward

            else:
                if _info['reward_touch_puck'] > 0:
                    touch_stats[episode_counter] = 1

                total_reward += reward

            if agent.args.show:
                time.sleep(0.01)
                env.render()
            if done:
                if evaluate_on_opposite_side:
                    won_stats[episode_counter] = 1 if env.winner == -1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == 1 else 0
                else:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                break

        rew_stats.append(total_reward)
        if not quiet:
            agent.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon=0,
                                            touched=touch_stats[episode_counter])

    if not quiet:
        # Print evaluation stats
        agent.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

    # Toggle the verbose flag onto the old value
    env.verbose = old_verbose

    return (
        np.mean(rew_stats),
        np.mean(list(touch_stats.values())),
        np.mean(list(won_stats.values())),
        np.mean(list(lost_stats.values()))
    )
