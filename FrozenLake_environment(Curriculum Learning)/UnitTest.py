# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
from typing import Callable
import os
import torch
import numpy as np
import utils as ut

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

############################################################################################################
import random
import string
from itertools import product

# Generate all possible 3-letter codes
codes = [''.join(letters) for letters in product(string.ascii_uppercase, repeat=3)]
random.shuffle(codes)  # Shuffle the list to ensure randomness

def get_unique_code():
    """Generates a unique 3-letter code."""
    if codes:
        return codes.pop()
    else:
        raise ValueError("All 3-letter codes have been used.")

# Example usage:
unique_code = get_unique_code()

############################################################################################################

# Define the curriculum stages with increasing hole percentages
curriculum = [0.01, 0.05, 0.1]   # Start with 1% holes, then 5%, then 10%.

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

#############################################################################################################
# Define the path for logging
log_path = os.path.join(f'Training_{unique_code}', 'Logs_curriculum_threshold')
os.makedirs(log_path, exist_ok=True)  # Ensure the directory exists

# Initialize the model variable (will be assigned during the first stage)
model = None

# Define the performance threshold based on mean reward
performance_threshold = 0.90  # Agent must achieve a mean reward of 0.90 to move to next stage
evaluation_episodes = 5       # Number of episodes to evaluate the agent
max_timesteps_per_stage = 500000  # Maximum timesteps to train per stage to prevent infinite loops

##############################################################################################################

# Custom callback to check performance and implement threshold
class ThresholdCallback(BaseCallback):
    def __init__(self, eval_env, threshold, eval_episodes, verbose=0):
        super(ThresholdCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.threshold = threshold
        self.eval_episodes = eval_episodes
        self.mean_reward = 0.0
        self.stage_timesteps = 0

    def _on_step(self) -> bool:
        # Increment by the number of timesteps since last call
        self.stage_timesteps += self.locals['n_steps']
        # Evaluate every 5000 timesteps
        if self.stage_timesteps % 1000 <= self.locals['n_steps']:
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.eval_episodes,
                render=False,
                deterministic=True,
                warn=False
            )
            self.mean_reward = mean_reward
            print(f"Evaluation at timestep {self.num_timesteps}: Mean Reward = {self.mean_reward}")
            if self.mean_reward >= self.threshold:
                print(f"Mean reward {self.mean_reward} >= threshold {self.threshold}. Moving to next stage.")
                return False  # Returning False stops training
        return True

##############################################################################################################

# Start the loop over stages
for stage, hole_percentage in enumerate(curriculum, 1):
    print(f"Stage {stage}, Hole Percentage: {hole_percentage}")

    # Generate the custom map
    custom_map = ut.generate_valid_map(size, hole_percentage)
    print(f"Custom Map for Stage {stage}:\n{custom_map}")

    # Create the training environment
    train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])

    # Create the evaluation environment
    eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Instantiate the threshold callback
    threshold_callback = ThresholdCallback(
        eval_env=eval_env,
        threshold=performance_threshold,
        eval_episodes=evaluation_episodes,
        verbose=1
    )

    if stage == 1:
        print(f"Starting a new model for Stage {stage}")
        # Modeling
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
            learning_rate=linear_schedule(3e-4)
        )
    else:
        print(f"Continuing training with the model from Stage {stage - 1}")
        # Update the environment in the existing model
        model.set_env(train_env)
        # Optionally, adjust learning rate or other hyperparameters here

    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=max_timesteps_per_stage, callback=threshold_callback)
    print(f"Training Completed for Stage {stage}")

    # Save the model
    PPO_Path = os.path.join(f'Training_{unique_code}', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')
    os.makedirs(os.path.dirname(PPO_Path), exist_ok=True)  # Ensure the directory exists
    print('Saving the model')
    model.save(PPO_Path)

print("Training across all stages completed.")
