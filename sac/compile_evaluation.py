# Run this function to compile a bunch of evaluation results into a single json file

from argparse import ArgumentParser
from laserhockey import hockey_env as h_env
from sac_agent import SACAgent
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils import *
from base.evaluator import evaluate

parser = ArgumentParser()

# Training params
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=10)
parser.add_argument('--max_steps', help='Set number of steps in an eval episode', type=int, default=10000)
parser.add_argument('--filename', help='Path to the pretrained model', default=None)
parser.add_argument('--mode', help='Mode for evaluation', default='normal')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--agents_folder', help='Folder with agents to evaluate', default=None)
parser.add_argument('--opponent_path', help='Folder with opponents to evaluate', default=None)

args = parser.parse_args()

if __name__ == '__main__':
    if args.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif args.mode == 'shooting':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif args.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help.')

    env = h_env.HockeyEnv(mode=mode)
    agents_folder = args.agents_folder

    json_file_name = f'sac/experiment.json'
    
    opponents = []
    opponents.append(('Strong Opponent' , h_env.BasicOpponent(weak=False)))
    opponents.append(('Weak Opponent', h_env.BasicOpponent(weak=True)))
    opponents.append(('Best Performing SAC', SACAgent.load_model_old(args.opponent_path)))

    agent_name = 'Phased SAC'
    
    episodes_dict = {}
    for episodes in range(1000, 21000, 1000):
        agent_num = f'a-{episodes}'
        agent_filename = f'{agents_folder}/agents/{agent_num}.pkl'
        agent = SACAgent.load_model_old(agent_filename)
        agent.eval()
        agent.args.max_steps = args.max_steps
        agent.args.q = args.q
        agent.args.eval_episodes = args.eval_episodes
        agent.args.mode = args.mode
        episodes_dict[episodes] =  {}
        for name, opponent in opponents:
            stats = evaluate(agent, env, opponent, args.eval_episodes)
            episodes_dict[episodes][name] = {name: stats, 'episodes': episodes}
    with open(json_file_name, 'w') as f:
        json.dump(episodes_dict, f, indent=4)

        
