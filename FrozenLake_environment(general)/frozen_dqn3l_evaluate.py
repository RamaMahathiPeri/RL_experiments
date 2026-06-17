#IMPORT DEPENDENCIES

import gymnasium as gym
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor,VecNormalize,VecFrameStack,VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback,CallbackList
from stable_baselines3.common.utils import get_linear_fn

from typing import Callable
import os 
import pygame
#pygame.init()
import torch

# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal) backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

environment_name = 'FrozenLake-v1'
DQN_Path = ('/Users/ramamahathiperi/Desktop/RL_Nick/03_FrozenLake-environment /Training/Saved Models_3l/FrozenLake_Dqn3l.zip')
#vec_normalize_stats_path= ('/Users/ramamahathiperi/Training/Saved Models_2l/vec_normalize_2l.pkl')

#/Users/ramamahathiperi/Training/Saved Models_2l

env = gym.make(environment_name,render_mode='human',desc=None,map_name='8x8',is_slippery='True')
env = DummyVecEnv([lambda: env])


model = DQN.load(DQN_Path,env=env,device="mps")




# Evaluation Environment
eval_env = gym.make(environment_name,render_mode='human',desc=None,map_name='8x8',is_slippery='True')
eval_env = DummyVecEnv([lambda: eval_env])



# Evaluation
print('Evaluating the model...')
episode_rewards, episode_lengths= evaluate_policy(model, eval_env, n_eval_episodes=10, render=True,return_episode_rewards=True)
print('Evaluation completed')

# Print the reward for each episode
for idx, reward in enumerate(episode_rewards):
    print(f"Episode {idx + 1}: Reward = {reward}")

eval_env.close()
