# Run this file to evaluate a single agent in multiple games and visualize the games

from argparse import ArgumentParser
from laserhockey import hockey_env as h_env
from sac_agent import SACAgent
import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils import *
from base.evaluator import evaluate

parser = ArgumentParser()

# Training params
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=30)
parser.add_argument('--max_steps', help='Set number of steps in an eval episode', type=int, default=250)
parser.add_argument('--filename', help='Path to the pretrained model', default=None)
parser.add_argument('--mode', help='Mode for evaluating currently: (shooting | defense)', default='shooting')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--opposite', help='Evaluate agent on opposite side', default=False, action='store_true')
parser.add_argument('--weak', help='Evaluate agent vs weak basic opponent', default=False, action='store_true')

opts = parser.parse_args()

if __name__ == '__main__':
    if opts.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif opts.mode == 'shooting':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help.')

    if opts.filename is None:
        raise ValueError('Parameter --filename must be present. See --help.')

    env = h_env.HockeyEnv(mode=mode)
    agent = SACAgent.load_model_old(opts.filename)
    agent.args.max_steps = opts.max_steps
    agent.args.q = opts.q
    agent.args.eval_episodes = opts.eval_episodes
    agent.args.mode = opts.mode
    agent.eval()
    agent.args.show = opts.show
    opponents = []
    opponent_name = 'sac/old_models/230808_142054/agents/a-18000.pkl'
    opponent = h_env.BasicOpponent(weak=opts.weak)
    #opponent = SACAgent.load_model_old(opponent_name) # 
    stats = evaluate(agent, env, opponent, opts.eval_episodes)
    print(stats)
