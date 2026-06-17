#IMPORT DEPENDENCIES

import gymnasium as gym
from stable_baselines3 import PPO,SAC
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

environment_name = 'CarRacing-v2'
PPO_Path = ('/Users/ramamahathiperi/Training/Saved Models_2l/Model_RacingCar_Cnn2l.zip')
vec_normalize_stats_path= ('/Users/ramamahathiperi/Training/Saved Models_2l/vec_normalize_2l.pkl')

#/Users/ramamahathiperi/Training/Saved Models_2l

env = gym.make(environment_name,render_mode='human')
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
env = VecMonitor(env)

model = PPO.load(PPO_Path,env=env,device="mps")


# Save the normalization statistics after training
env.save(vec_normalize_stats_path)

# Evaluation Environment
eval_env = gym.make(environment_name,render_mode='human')
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

# Load the saved VecNormalize statistics
eval_env = VecNormalize.load(vec_normalize_stats_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False
eval_env = VecMonitor(eval_env)

# Evaluation
print('Evaluating the model...')
evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print('Evaluation completed')

eval_env.close()
