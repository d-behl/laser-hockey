import numpy as np
import torch
import io
from pathlib import Path
import pickle
import os
import sys
# sys.path.append(os.getcwd())
sys.path.insert(0, '.')
sys.path.insert(1, '..')
from dqn.agent import DQNAgent
# print(os.getcwd())


from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
# from remoteControllerInterface  import RemoteControllerInterface
from client.backend.client import Client
# from backend.client import Client

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_model(filename):

    load_path = Path(filename)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if device == torch.device('cpu'):
    #     return torch.load(load_path, map_location=torch.device('cpu'))
    with open(load_path, 'rb') as inp:
        # return pickle.load(inp)
        if device == torch.device('cpu'):
            return CPU_Unpickler(inp).load()
        else:
            return pickle.load(inp)

class RemoteMyOpponent(RemoteControllerInterface):

    def __init__(self, filename):
        RemoteControllerInterface.__init__(self, identifier='RNBW')
        self.q_agent = load_model(filename)
        self.q_agent.eval()
        

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        a1 = self.q_agent.act(obs, eps=0)
        return np.asarray(self.q_agent.action_mapping[a1])
    
if __name__ == '__main__':
    filename = 'RNBW_EVEN_CMPLX_OPPS_NOISY_50000.pkl'
    controller = RemoteMyOpponent(filename)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='What the Puck?:RNBW', # Testuser
                    password='waeVae4ohd',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/RNBW', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=1000)
