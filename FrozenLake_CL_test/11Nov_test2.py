# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from typing import Callable
import os
import torch
import numpy as np
import utils as ut

print(os.getcwd())
# Define the path for logging
log_path = os.path.join('training_flt_4', 'Logs_curriculum_threshold')

# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal) backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

print(f"Current working directory: {os.getcwd()}")

# ENVIRONMENT PARAMETERS
environment_name = 'FrozenLake-v1'
size = 4
PPO_Path= os.path.join('training_flt_4','saved_models_stage_1','frozenlake_stage_1.zip')
hole_percentage=0.1
custom_map= ut.generate_valid_map(size,hole_percentage)
#print the custom map 
print(f'printing the custom map for stage 3')
ut.print_map(custom_map)
#creating the training env 
train_env=gym.make(environment_name,desc= custom_map,is_slippery=True,render_mode=None)
train_env=Monitor(train_env)
train_env= DummyVecEnv([lambda:train_env])
#load the model
model= PPO.load(PPO_Path,env=train_env,device=device)

# Define the evaluation environment
eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

# Define the callback to stop training when average reward reaches 0.9
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)

# Define the EvalCallback
eval_callback = EvalCallback(eval_env,
                             callback_on_new_best=stop_callback,
                             eval_freq=1000,
                             n_eval_episodes=10,
                             best_model_save_path='./best_model',
                             verbose=1)

# Train the model with the callback
print("Training the model for Stage 3")
model.learn(total_timesteps=200000, callback=eval_callback)
print("Training Completed for Stage 3 ")

# Save the final model
PPO_Path = os.path.join('training_flt_4', 'Saved_Models_Stage_4', 'FrozenLake_Stage_4')
print('Saving the model')
model.save(PPO_Path)

# Load the best model saved during training
model = PPO.load(os.path.join('./best_model', 'best_model'), env=eval_env, device=device)

# Evaluation
print('Evaluating the model...')
episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True, return_episode_rewards=True)
print('Evaluation completed')

# Print the reward for each episode
for idx, reward in enumerate(episode_rewards):
    print(f"Episode {idx + 1}: Reward = {reward}")

eval_env.close()
