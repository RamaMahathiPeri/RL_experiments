# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold  # Added StopTrainingOnRewardThreshold
)
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
curriculum = [0.01, 0.05, 0.1]  # Start with 1% holes, then 5%, then 10%.

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# Define the path for logging
log_path = os.path.join('Training', 'Logs_curriculum_threshold_2')

# Define the reward threshold
reward_threshold = 0.9  # Set your desired threshold here

for stage, hole_percentage in enumerate(curriculum, 1):
    if stage == 1:
        print('We are in Stage 1: Starting a new model')
        PPO_Path = os.path.join('training_flt_5', f'saved_models_stage_{stage}', f'frozenlake_stage_{stage}.zip')
        hole_percentage = curriculum[stage - 1]  # Corrected index
        custom_map = ut.generate_valid_map(size, hole_percentage)
        # Print the custom map
        print(f'Printing the custom map for Stage {stage}')
        ut.print_map(custom_map)
        # Creating the training environment
        train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
        train_env = Monitor(train_env)
        train_env = DummyVecEnv([lambda: train_env])
        # Initialize the model
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,  # Use the device determined earlier
            learning_rate=linear_schedule(3e-4)
        )
    else:
        print(f'We are in Stage {stage}')
        PPO_Path = os.path.join('training_flt_5', f'saved_models_stage_{stage - 1}', f'frozenlake_stage_{stage - 1}.zip')
        print('######')
        print(PPO_Path)
        print('########')
        hole_percentage = curriculum[stage - 1]  # Corrected index
        custom_map = ut.generate_valid_map(size, hole_percentage)
        # Print the custom map
        print(f'Printing the custom map for Stage {stage}')
        ut.print_map(custom_map)
        # Creating the training environment
        train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
        train_env = Monitor(train_env)
        train_env = DummyVecEnv([lambda: train_env])
        # Load the model
        model = PPO.load(PPO_Path, env=train_env, device=device)

    # Create the evaluation environment without rendering
    eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Set up the StopTrainingOnRewardThreshold callback
    stop_train_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )

    # Set up the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_train_callback,
        best_model_save_path=f'./logs_stage_{stage}/',
        log_path=f'./logs_stage_{stage}/',
        eval_freq=1000,  # Evaluate every 1000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,  # Number of episodes to evaluate
        verbose=1
    )

    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=50000, callback=eval_callback)
    print(f"Training Completed for Stage {stage}")

    # Save the model
    PPO_Path = os.path.join('training_flt_5', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')
    print('Saving the model')
    model.save(PPO_Path)
