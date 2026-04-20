# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
import os
import torch
import numpy as np
import utils as ut
from typing import Callable

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
size = 6

# Define the curriculum stages with increasing hole percentages
curriculum = [0.05, 0.1, 0.2, 0.3]  # Start with 5%, then 10%, 20%, and 30%.

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

unique_code = ut.get_unique_code()

# Define the path for logging
log_path = os.path.join(f'Training_1_{unique_code}', 'Logs_curriculum_threshold')

# Custom Environment Wrapper
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

class ChangingMapFrozenLakeEnv(gym.Wrapper):
    def __init__(self, env, size=6, hole_percentage=0.05, is_slippery=True, change_interval=50):
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
    PPO_Path = os.path.join(f'Training_1_{unique_code}', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')

    # Create the base environment
    base_env = gym.make(environment_name, is_slippery=True, render_mode=None)

    # Wrap the environment to change the map every 50 steps
    train_env = ChangingMapFrozenLakeEnv(
        base_env,
        size=size,
        hole_percentage=hole_percentage,
        is_slippery=True,
        change_interval=1000
    )
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])

    # Create the evaluation environment without rendering
    eval_env_base = gym.make(environment_name, is_slippery=True, render_mode=None)
    eval_env = ChangingMapFrozenLakeEnv(
        eval_env_base,
        size=size,
        hole_percentage=hole_percentage,
        is_slippery=True,
        change_interval=50
    )
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Initialize or load the model
    if stage == 1:
        print('Initializing a new model.')
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
            learning_rate=linear_schedule(3e-4)
        )
    else:
        prev_PPO_Path = os.path.join(f'Training_1_{unique_code}', f'Saved_Models_Stage_{stage-1}', f'FrozenLake_Stage_{stage-1}')
        print(f'Loading model from {prev_PPO_Path}')
        model = PPO.load(prev_PPO_Path, env=train_env, device=device)

    # Define the callback to stop training when average reward reaches 0.9
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)

    # Define the EvalCallback
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=100,
                                 n_eval_episodes=10,
                                 best_model_save_path='./best_model',
                                 verbose=1)

    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=10000000, callback=eval_callback)
    print(f"Training Completed for Stage {stage}")

    # Save the model
    os.makedirs(os.path.dirname(PPO_Path), exist_ok=True)  # Ensure the directory exists
    print('Saving the model')
    model.save(PPO_Path)

    # Load the best model saved during training
    model = PPO.load(PPO_Path, env=eval_env, device=device)

    # Evaluation
    print('Evaluating the model...')
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False, return_episode_rewards=True)
    print('Evaluation completed')

    # Print the reward for each episode
    for idx, reward in enumerate(episode_rewards):
        print(f"Episode {idx + 1}: Reward = {reward}")

    eval_env.close()
