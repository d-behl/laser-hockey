import numpy as np
import torch
import io
from pathlib import Path
import pickle
import os
import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

from sac.sac_agent import SACAgent

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_model(filename):

    load_path = Path(filename)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(load_path, 'rb') as inp:
        if device == torch.device('cpu'):
            return CPU_Unpickler(inp).load()
        else:
            return pickle.load(inp)

class RemoteMyOpponent(RemoteControllerInterface):

    def __init__(self, filename, identifier='SAC'):
        RemoteControllerInterface.__init__(self, identifier=identifier)
        self.q_agent = SACAgent.load_model_old(filename)
        self.q_agent.eval()
        

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        a1 = self.q_agent.act(obs)
        return np.asarray(a1)
    
if __name__ == '__main__':
    filename = 'sac/models/230808_213414/agents/a-3000.pkl'

    controller = RemoteMyOpponent(filename)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='What the Puck?:', # Testuser
                    password='waeVae4ohd',
                    controller=controller, 
                    output_path='/home/bozcomlekci/Desktop/ML/RL/laser-hockey/RL2023HockeyTournamentClient/serverside', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=1)