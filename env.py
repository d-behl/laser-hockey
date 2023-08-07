# %%
import numpy as np
import laserhockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time

# %%
np.set_printoptions(suppress=True)

# %%
reload(h_env)

# %% [markdown]
# # Normal Game Play

# %%
env = h_env.HockeyEnv()

# %% [markdown]
# have a look at the initialization condition: alternating who starts and are random in puck position

# %%
obs, info = env.reset()
obs_agent2 = env.obs_agent_two()
_ = env.render()

# %% [markdown]
# one episode with random agents

# %%
obs, info = env.reset()
obs_agent2 = env.obs_agent_two()

for _ in range(600):
    env.render(mode="human")
    a1 = np.random.uniform(-1,1,4)
    a2 = np.random.uniform(-1,1,4)    
    obs, r, d, t, info = env.step(np.hstack([a1,a2]))    
    obs_agent2 = env.obs_agent_two()
    if d: break

# %% [markdown]
# Without rendering, it runs much faster

# %% [markdown]
# "info" dict contains useful proxy rewards and winning information

# %%
info

# %% [markdown]
# Winner == 0: draw
# 
# Winner == 1: you (left player)
# 
# Winner == -1: opponent wins (right player)

# %%
env.close()

# %% [markdown]
# # Train Shooting

# %%
env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)

# %%
o, info = env.reset()
_ = env.render()

for _ in range(500):
    env.render()
    a1 = [1,0,0,1] # np.random.uniform(-1,1,4)
    a2 = [0,0.,0,0] 
    obs, r, d, _ , info = env.step(np.hstack([a1,a2]))    
    obs_agent2 = env.obs_agent_two()
    if d: break

# %%


# %%
env.close()

# %% [markdown]
# # Train DEFENDING

# %%
env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

# %%
o, info = env.reset()
_ = env.render()

for _ in range(60):
    env.render()
    a1 = [0.1,0,0,1] # np.random.uniform(-1,1,3)
    a2 = [0,0.,0,0] 
    obs, r, d,_, info = env.step(np.hstack([a1,a2]))
    print(r)
    obs_agent2 = env.obs_agent_two()
    if d: break

# %%
env.close()

# %% [markdown]
# # Using discrete actions

# %%
import random

# %%
env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)

# %%
env.reset()
for _ in range(200):
    env.render()
    a1_discrete = random.randint(0,7)
    a1 = env.discrete_to_continous_action(a1_discrete)
    a2 = [0,0.,0,0 ] 
    obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
    obs_agent2 = env.obs_agent_two()
    if d: break

# %%
env.close()

# %% [markdown]
# # Hand-crafted Opponent

# %%
env = h_env.HockeyEnv()

# %%
o, info = env.reset()
_ = env.render()
player1 = h_env.BasicOpponent(weak=False)
player2 = h_env.BasicOpponent()

# %%
obs_buffer = []
reward_buffer=[]
obs, info = env.reset()
obs_agent2 = env.obs_agent_two()
for _ in range(250):
    env.render()
    a1 = player1.act(obs)
    a2 = player2.act(obs_agent2)
    obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
    obs_buffer.append(obs)
    reward_buffer.append(r)
    obs_agent2 = env.obs_agent_two()
    if d: break
obs_buffer = np.asarray(obs_buffer)
reward_buffer = np.asarray(reward_buffer)

# %%
np.mean(obs_buffer,axis=0)

# %%
np.std(obs_buffer,axis=0)

# %% [markdown]
# If you want to use a fixed observation scaling, this might be a reasonable choice

# %%
scaling = [ 1.0,  1.0 , 0.5, 4.0, 4.0, 4.0,  
            1.0,  1.0,  0.5, 4.0, 4.0, 4.0,  
            2.0, 2.0, 10.0, 10.0, 4,0 ,4,0]

# %%
import pylab as plt

# %%
plt.plot(obs_buffer[:,2])
plt.plot(obs_buffer[:,8])

# %%
plt.plot(obs_buffer[:,12])

# %%
plt.plot(reward_buffer[:])

# %%
np.sum(reward_buffer)

# %%
env.close()

# %% [markdown]
# # Human Opponent

# %%
env = h_env.HockeyEnv()

# %%
player1 = h_env.HumanOpponent(env=env, player=1)
player2 = h_env.BasicOpponent()


# %%
player1 = h_env.BasicOpponent()
player2 = h_env.HumanOpponent(env=env, player=2)


# %%
obs, info = env.reset()

env.render()
time.sleep(1)
obs_agent2 = env.obs_agent_two()
while True:
    time.sleep(0.2)
    env.render()
    a1 = player1.act(obs) 
    a2 = player2.act(obs_agent2)
    obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
    obs_agent2 = env.obs_agent_two()
    if d: break

# %%
env.close()


