# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import torch
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

# Curriculum stages with increasing hole percentages
curriculum = [0.01, 0.05, 0.1]  # Start with 1% holes, then 5%, then 10%.

# Define a linear learning rate schedule
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

# Define the path for logging
log_path = os.path.join('Training_12Nov', 'Logs_curriculum_threshold')

# Initialize the model variable
model = None

# Custom Environment Wrapper
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

class ChangingMapFrozenLakeEnv(gym.Wrapper):
    def __init__(self, env, size=4, hole_percentage=0.1, is_slippery=True, change_interval=50):
        super(ChangingMapFrozenLakeEnv, self).__init__(env)
        self.size = size
        self.hole_percentage = hole_percentage
        self.is_slippery = is_slippery  # Store is_slippery
        self.change_interval = change_interval
        self.step_count = 0

    def reset(self, **kwargs):
        # Reset the step count
        self.step_count = 0
        # Generate a new map
        custom_map = ut.generate_valid_map(self.size, self.hole_percentage)
        # Update the environment's map
        self.env = FrozenLakeEnv(desc=custom_map, is_slippery=self.is_slippery)
        self.env.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        if self.step_count >= self.change_interval:
            # Reset the environment with a new map
            observation = self.reset()
            reward = 0
            terminated = True
            truncated = False
            info = {}
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

# Training loop for curriculum stages
for stage, hole_percentage in enumerate(curriculum, 1):
    print(f"\n=== Stage {stage}: Hole Percentage {hole_percentage*100}% ===")

    # Create the base environment
    base_env = gym.make(environment_name, is_slippery=True, render_mode=None)

    # Wrap the environment to change the map every 50 steps
    train_env = ChangingMapFrozenLakeEnv(
        base_env,
        size=size,
        hole_percentage=hole_percentage,
        is_slippery=True,           # Pass is_slippery here
        change_interval=50
    )
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])

    # Initialize or update the model
    if model is None:
        print("Initializing a new model.")
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
            learning_rate=linear_schedule(3e-4)
        )
    else:
        print("Updating the model's environment.")
        model.set_env(train_env)

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=500000)
    print("Training Completed.")

    # Save the model
    PPO_Path = os.path.join('Training_12Nov', 'Saved_Models_1', f'FrozenLake_stage_{stage}')
    print(f"Saving the model to {PPO_Path}")
    model.save(PPO_Path)
