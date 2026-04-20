# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
from typing import Callable
import os
import torch
import numpy as np
import utils as ut

print(os.getcwd())

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

# Define the curriculum stages with increasing hole percentages
curriculum = [0.05, 0.1, 0.2, 0.3]  # Start with 5%, then 10%,20 and 30%.

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

unique_code = ut.get_unique_code()



# Define the path for logging
log_path = os.path.join(f'Training_{unique_code}', 'Logs_curriculum_threshold')

for stage,hole_percentage in enumerate(curriculum,1):
    if stage == 1:
        print('we are in stage 1: starting a new model')
        PPO_Path= os.path.join(f'Training_{unique_code}',f'saved_models_stage_{stage}',f'frozenlake_stage_{stage}.zip')
        hole_percentage=curriculum[stage]
        custom_map= ut.generate_valid_map(size,hole_percentage)
        #print the custom map 
        print(f'printing the custom map for stage {stage}')
        ut.print_map(custom_map)
        #creating the training env 
        train_env=gym.make(environment_name,desc= custom_map,is_slippery=True,render_mode=None)
        train_env=Monitor(train_env)
        train_env= DummyVecEnv([lambda:train_env])
        #load the model
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,  # Use the device determined earlier
            learning_rate=linear_schedule(3e-4)
        )
    else:
        print(f'we are in stage{stage}')
        PPO_Path= os.path.join(f'Training_{unique_code}',f'saved_models_stage_{stage-1}',f'frozenlake_stage_{stage-1}.zip')
        print('######')
        print(PPO_Path)
        print('########')
        hole_percentage=curriculum[stage-1]
        custom_map= ut.generate_valid_map(size,hole_percentage)
        #print the custom map 
        print(f'printing the custom map for stage {stage}')
        ut.print_map(custom_map)
        #creating the training env 
        train_env=gym.make(environment_name,desc= custom_map,is_slippery=True,render_mode=None)
        train_env=Monitor(train_env)
        train_env= DummyVecEnv([lambda:train_env])
        #load the model
        model= PPO.load(PPO_Path,env=train_env,device=device)
    
    # Create the evaluation environment without rendering
    eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=500000)
    print(f"Training Completed for Stage {stage}")

    # Save the model
    PPO_Path = os.path.join(f'Training_{unique_code}', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')
    os.makedirs(os.path.dirname(PPO_Path), exist_ok=True)  # Ensure the directory exists
    print('Saving the model')
    model.save(PPO_Path)

    


