# %%
import numpy as np
import laserhockey.laser_hockey_env as lh
import gymnasium as gym
from importlib import reload

# # %%
np.set_printoptions(suppress=True)

# # %% [markdown]
# # # Normal Game Play

# # %%
# reload(lh)

# # %%
# env = lh.LaserHockeyEnv()

# # %% [markdown]
# # have a look at the initialization condition: alternating who starts and are random in puck position

# # %%
# obs, info = env.reset()
# obs_agent2 = env.obs_agent_two()
# env.render()

# # %% [markdown]
# # one episode with random agents

# # %%
# #obs = env.reset()
# obs_agent2 = env.obs_agent_two()

# for _ in range(600):
#     env.render()
#     a1 = np.random.uniform(-1,1,3)
#     a2 = np.random.uniform(-1,1,3)    
#     obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
#     obs_agent2 = env.obs_agent_two()
#     if d: break

# # %% [markdown]
# # Without rendering, it runs much faster

# # %%
# obs, info  = env.reset()
# obs_agent2 = env.obs_agent_two()

# for _ in range(600):    
#     a1 = [1,-.5,0] # np.random.uniform(-1,1,3)
#     a2 = [1,0.,0] # np.random.uniform(-1,1,3)*0    
#     obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
#     obs_agent2 = env.obs_agent_two()
#     if d: break

# # %% [markdown]
# # "info" dict contains useful proxy rewards and winning information

# # %%
# info

# # %% [markdown]
# # Winner == 0: draw
# # 
# # Winner == 1: you (left player)
# # 
# # Winner == -1: opponent wins (right player)

# # %% [markdown]
# # # Train Shooting

# # %%
# env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_SHOOTING)

# # %%
# o, info = env.reset()
# env.render()

# # %%
# for _ in range(200):
#     env.render()
#     a1 = [1,0,0] # np.random.uniform(-1,1,3)
#     a2 = [0,0.,0] 
#     obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
#     obs_agent2 = env.obs_agent_two()
#     if d: break

# # %% [markdown]
# # # Train DEFENDING

# # %%
# reload(lh)

# # %%
# env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_DEFENSE)

# # %%
# o, info = env.reset()
# env.render()

# # %%
# for _ in range(60):
#     env.render()
#     a1 = [1,0,0] # np.random.uniform(-1,1,3)
#     a2 = [0,0.,0] 
#     obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
#     obs_agent2 = env.obs_agent_two()
#     if d: break

# # %% [markdown]
# # # Using discrete actions

# # %%
# env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_SHOOTING)

# # %%
# import random

# # %%
# for _ in range(200):
#     env.render()
#     a1_discrete = random.randint(0,7)
#     a1 = env.discrete_to_continous_action(a1_discrete)
#     a2 = [0,0.,0] 
#     obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
#     obs_agent2 = env.obs_agent_two()
#     if d: break

# # %%


# # %% [markdown]
# # # Hand-crafted Opponent

# # %%
# reload(lh)

# # %%
# env = lh.LaserHockeyEnv()

# # %%
# o, info = env.reset()
# env.render()

# # %%
# player1 = lh.BasicOpponent()
# player2 = lh.BasicOpponent()

# # %%
# obs_buffer = []

# # %%
# obs, info = env.reset()
# obs_agent2 = env.obs_agent_two()
# for _ in range(100):
#     env.render()
#     a1 = player1.act(obs)
#     a2 = player2.act(obs_agent2)
#     obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
#     obs_buffer.append(obs)
#     obs_agent2 = env.obs_agent_two()
#     if d: break

# # %%
# obs_buffer = np.asarray(obs_buffer)

# # %%
# np.mean(obs_buffer,axis=0)

# # %%
# np.std(obs_buffer,axis=0)

# # %%
# scaling = [ 1.0,  1.0 , 3.14, 4.0, 4.0, 2.0,  
#             1.0,  1.0,  3.14, 4.0, 4.0, 2.0,  
#             2.0, 2.0, 10.0, 10.0]

# # %%
# env.close()

# %% [markdown]
# # Human Opponent

# %%
import time

# %%
env = lh.LaserHockeyEnv()

# %%
o, info = env.reset()
env.render()

# %%
from laserhockey.hockey_env import HumanOpponent

# %%
player1 = HumanOpponent(env=env, player=1)
player2 = lh.BasicOpponent()

# %%
obs, info = env.reset()
time.sleep(2)
obs_agent2 = env.obs_agent_two()
while True:
    env.render()
    a1 = player1.act(obs)
    a2 = player2.act(obs_agent2)
    obs, r, d, _, info = env.step(np.hstack([a1,a2]))
    print(r, info, env.action_space)
    time.sleep(0.01)
    obs_agent2 = env.obs_agent_two()
    if d: break

# %%



