import os
import torch
from laserhockey import hockey_env as h_env
from sac_agent import SACAgent
from argparse import ArgumentParser
import sys
from trainer import SACTrainer
import time
import random
import json 

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils import Logger
from base.experience_replay import ExperienceReplay

parser = ArgumentParser()
parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true')
parser.add_argument('--cuda', help='Set if want to train on graphic card', action='store_true')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='defense')
parser.add_argument('--preload_path', help='Path to the pretrained model', default=None)
parser.add_argument('--transitions_path', help='Path to the root of folder containing transitions', default=None)

# Training params
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=5000)
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=250)
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=30)
parser.add_argument('--evaluate_every',
                    help='# of episodes between evaluating agent during the training', type=int, default=1000)
parser.add_argument('--add_self_every',
                    help='# of gradient updates between adding agent (self) to opponent list', type=int, default=100000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--alpha_lr', help='Learning rate', type=float, default=1e-4)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--alpha_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--update_target_every', help='# of steps between updating target net', type=int, default=1)
parser.add_argument('--gamma', help='Discount', type=float, default=0.95)
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--grad_steps', help='grad_steps', type=int, default=32)
parser.add_argument(
    '--alpha',
    type=float,
    default=0.2,
    help='Temperature parameter alpha determines the relative importance of the entropy term against the reward')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False,
                    help='Automatically adjust alpha')
parser.add_argument('--selfplay', type=bool, default=False, help='Should agent train selfplaf')
parser.add_argument('--soft_tau', help='tau', type=float, default=0.005)
parser.add_argument('--per', help='Utilize Prioritized Experience Replay', action='store_true')
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)
parser.add_argument('--phased', help='Phased training', type=str, default=None)
parser.add_argument('--extra_info', help='Add extra info to observation', action='store_true')
parser.add_argument('--opposite_side', help='Train agent on opposite side', action='store_true')
parser.add_argument('--noise', help='Add noise to observation', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dry_run:
        args.max_episodes = 10

    if args.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif args.mode == 'shooting':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif args.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    os.makedirs('models', exist_ok=True)
    dirname = time.strftime(f'models/%y%m%d_%H%M%S', time.gmtime(time.time()))
    abs_path = os.path.dirname(os.path.realpath(__file__))
    logger = Logger(prefix_path=os.path.join(abs_path, dirname),
                    mode=args.mode,
                    cleanup=True,
                    quiet=args.q)
    
    with open(os.path.join(abs_path, dirname, 'args.json'), 'w') as f:
        json.dump(str(args), f, indent=4)

    env = h_env.HockeyEnv(mode=mode, verbose=(not args.q))
    opponents = [
        h_env.BasicOpponent(weak=True),
        h_env.BasicOpponent(weak=False)
    ]

    # Add absolute paths for pretrained agents
    pretrained_agents = []

    if args.selfplay:
        for p in pretrained_agents:
            a = SACAgent.load_model(p)
            a.eval()
            opponents.append(a)
    if args.extra_info:
        # extend observation space by 1
        obs_dim = (env.observation_space.shape[0] + 1, )
    else:
        obs_dim = env.observation_space.shape  
    agent = SACAgent(
        logger=logger,
        obs_dim=obs_dim,
        action_space=env.action_space,
        action_dim=env.num_actions,
        args=args
    )
    if args.preload_path is not None:
        #agent.load_model(agent, args.preload_path)
        agent = SACAgent.load_model_old(args.preload_path)
        agent.args = args
        #agent.buffer.preload_transitions(args.transitions_path)
    agent.train()

    trainer = SACTrainer(logger, args=args)
    trainer.train(agent, opponents, env)
