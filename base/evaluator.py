import time
import numpy as np

from sac.utils import calculate_phase

def evaluate(agent, env, opponent, eval_episodes, quiet=False, action_mapping=None):
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
            phase = calculate_phase(obs=ob, env=env, info=None, player=1)
            # SAC act
            if agent.args.phased:
                a1 = agent.act(ob, phase=phase)
            else:
                a1 = agent.act(ob)

            if agent.args.mode in ['defense', 'normal']:
                a2 = opponent.act(obs_agent2)
            elif agent.args.mode == 'shooting':
                a2 = [0, 0, 0, 0]

            (ob_new, reward, done, _, _info) = env.step(np.hstack([a1, a2]))
            ob = ob_new
            obs_agent2 = env.obs_agent_two()

            if _info['reward_touch_puck'] > 0:
                touch_stats[episode_counter] = 1
            total_reward += reward

            if agent.args.show:
                time.sleep(0.01)
                env.render()
            if done:
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
