import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from pink import PinkActionNoise
import laserhockey.hockey_env as h_env
from importlib import reload
from gymnasium.wrappers import RecordVideo

import utils
import TD3
from memory.buffer import PrioritizedReplayBuffer

reload(h_env)

np.random.seed(15)
torch.manual_seed(15)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, render_mode=None, weak=True):
	eval_env = h_env.HockeyEnv()
	eval_env.reset()
	player2 = h_env.BasicOpponent(weak=weak)
	#RecordVideo(eval_env, './video/')
	eval_env = gym.wrappers.RecordVideo(eval_env, 'video')
	avg_reward = 0.
	wins = 0
	for i in range(eval_episodes):
		#state, info = eval_env.reset()
		truncated = False
		done = False
		obs, info = env.reset()
		obs_agent2 = env.obs_agent_two()
		while done:
			#time.sleep(0.2)
			a1 = policy.select_action(np.array(obs))
			a2 = player2.act(obs_agent2)
			obs, r, done, t , info = eval_env.step(np.hstack([a1,a2]))
			obs_agent2 = eval_env.obs_agent_two()
			if done or t: break
		wins += (info['winner'] == 1)

		#policy.action_noise.reset()
	#avg_reward /= eval_episodes
	eval_env.close()
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {wins/eval_episodes:.3f}")
	print("---------------------------------------")
	return wins/eval_episodes


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e7, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--priority", action="store_true")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	#env = gym.make(args.env)
	#env = lh.LaserHockeyEnv()
	env = h_env.HockeyEnv()

	# Set seeds
	env.reset()
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]//2
	max_action = float(env.action_space.high[0])

	noise_scale = 0.3
	seq_len = 1000
	action_noise = PinkActionNoise(noise_scale, seq_len, action_dim)

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"action_noise": action_noise,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	#elif args.policy == "OurDDPG":
#		policy = OurDDPG.DDPG(**kwargs)
	#elif args.policy == "DDPG":
	#	policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	# Initialize replay buffer
	if args.priority:
		replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size=int(1e5))
	else:
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]
	state, info = env.reset()
	#env.render()
	done = False
	truncated = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	touched = 0
	first_time_touch = 1
	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
			action[3:] = 0.
		else:
			action1 = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)
			action2 = [0. ,0., 0.]
			action = np.hstack((action1, action2))

		# Perform action
		next_state, reward, done, truncated, info = env.step(action)
		first_time_touch = 1 - touched
		#done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		done_bool = truncated or done

		# Store data in replay buffer
		replay_buffer.add(state, action[:3], next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)
		
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			evaluations.append(episode_reward)
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, info = env.reset()
			done = False
			truncated = False

			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			touched = 1
			first_time_touch = 0
			policy.action_noise.reset()
