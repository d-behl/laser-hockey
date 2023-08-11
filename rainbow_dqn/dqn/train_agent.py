import os
import torch
import laserhockey.hockey_env as h_env
from agent import DQNAgent
from argparse import ArgumentParser
import sys
from custom_action_space import DEFAULT_DISCRETE_ACTIONS, REDUCED_CUSTOM_DISCRETE_ACTIONS
from trainer import DQNTrainer

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils.utils import *

parser = ArgumentParser()
parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true', default=False)
## This is incorrect way of doing this, because there is no way of setting it to False
## But I set it here like this, because the command line was becoming too long
## Later all such options will be changed to store_false or maybe be given two options
parser.add_argument('--cuda', help='Set if want to train on graphic card', action='store_true', default=True)
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='normal')
parser.add_argument('--save_prefix', help='The saved model and plots have this prefix to identify the build', type=str, default='RAINBOW')

## If curriculm is true the 'mode' argument will be ignored
parser.add_argument('--curriculum', help='Enable curriculum learning', action='store_true', default=True)
parser.add_argument('--num_repeats', help='Number of times shooting and defense once chosen are repeated', default=10)

# Training params
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=80000)
parser.add_argument('--per_beta_inc', help='Beta increment for PER', type=float, default=0.000009)
parser.add_argument('--self_play', help='Utilize self play', action='store_true', default=False) #True)
parser.add_argument('--start_self_play_from', help='# of episode to start self play from', type=int, default=40000)
parser.add_argument('--add_opponent_every', help='# of grad updates until copying ourself', type=int, default=120000)

parser.add_argument('--max_steps', help='Max steps for training', type=int, default=250)
parser.add_argument('--start_learning_from', help='# of steps from which on learning happens', type=int, default=50000)
parser.add_argument('--train_every', help='Train every # of steps', type=int, default=10)
parser.add_argument('--update_target_every', help='# of gradient updates to updating target', type=int, default=1000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.0005)
parser.add_argument('--change_lr_every', help='Change learning rate every # of episodes', type=int, default=2500) #1500)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.75)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true', default=True)
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=1000)
parser.add_argument('--evaluate_every', help='Evaluate every # of episodes', type=int, default=2000)
parser.add_argument('--discount', help='Discount', type=float, default=0.99)
parser.add_argument('--epsilon', help='Epsilon', type=float, default=1)
parser.add_argument('--epsilon_decay', help='Epsilon decay', type=float, default=0.0005)
parser.add_argument('--min_epsilon', help='min_epsilon', type=float, default=0.1)
parser.add_argument('--dueling', help='Specifies whether the architecture should be dueling', action='store_true', default=True)
parser.add_argument('--double', help='Calculate target with Double DQN', action='store_true', default=True)
parser.add_argument('--per', help='Utilize Prioritized Experience Replay (PER)', action='store_true', default=True)
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)
parser.add_argument('--per_beta', help='Beta for PER', type=float, default=0.4)
parser.add_argument('--per_beta_max', help='Max beta for PER', type=float, default=1)
parser.add_argument('--multi_step', help='Enable multistep loss', action='store_true', default=True)
parser.add_argument('--n_steps', help='Number of steps for multistep RL', type=int, default=5)
parser.add_argument('--proximal_loss', help='Enable Proximal Loss', action='store_true', default=False) #This is the loss from paper fast RL with Slow updates
parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
parser.add_argument('--buffer_size', help='Buffer capacity for the experience replay', type=int, default=int(1e6))
parser.add_argument('--spectral_norm', help='Enable spectral norm', action='store_true', default=False)
parser.add_argument('--weight_norm', help='Enable spectral norm', action='store_true', default=False)
parser.add_argument('--NoisyNets', help='Enable NoisyNets', action='store_true', default=False)
parser.add_argument('--load_agents', help='Path to load agents', nargs='+', default='RNBW_CMPLX_REWARD_65000.pkl RNBW_CMPLX_RESET_72000.pkl RNBW_CMPLX_BOTH_70000.pkl') # RNBW_ADDED_OPP_NO_NOISY_72000.pkl RNBW_ADDED_OPP_EVAL_1000_66000.pkl')
parser.add_argument("--reset_branch", help='Enable resetting last branch', action='store_true', default=False)
parser.add_argument("--reset_last_layer", help='Enable resetting last layer only', action='store_true', default=False)
parser.add_argument("--reset_wait_eps", help='Number of episodes to train before next reset/player addition', type=int, default=20000)
opts = parser.parse_args()

if __name__ == '__main__':
    if opts.dry_run:
        opts.max_episodes = 2002

    ## MODE IS NOW USED INSIDE TRAINER.TRAIN HERE WE JUST DECLARE
    ## BASIC ENV TO USE IT LATER
    # if opts.mode == 'normal':
    #     mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    # elif opts.mode == 'shooting':
    #     mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    # elif opts.mode == 'defense':
    #     mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    # else:
    #     raise ValueError('Unknown training mode. See --help')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    opts.device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')
    print('Device: ', opts.device)
    print('Spectral: ', opts.spectral_norm)
    logger = Logger(prefix_path=os.path.dirname(os.path.realpath(__file__)) + '/logs_'+ opts.save_prefix,
                    mode=opts.mode,
                    cleanup=True,
                    quiet=opts.q)

    loaded_agents = []
    if opts.load_agents:
        for filename in opts.load_agents.split(' '):
            loaded_agents.append(logger.load_model(filename))

    env = h_env.HockeyEnv()#(not opts.q))

    action_mapping = REDUCED_CUSTOM_DISCRETE_ACTIONS

    q_agent = DQNAgent(
        logger=logger,
        obs_dim=env.observation_space.shape[0],
        action_mapping=action_mapping,
        userconfig=vars(opts)
    )
    trainer = DQNTrainer(logger=logger, config=vars(opts))
    print(q_agent.Q)
    trainer.train(agent=q_agent, env=env, loaded_agents=loaded_agents)
