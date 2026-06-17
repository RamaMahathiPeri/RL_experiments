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
log_path = os.path.join('Training_008', 'Logs_curriculum_threshold')

# Initialize the model variable (will be assigned during the first stage)
model = None

# Define the performance threshold
performance_threshold = 0.90  # Agent must achieve 90% success rate to move to next stage
evaluation_episodes = 5      # Number of episodes to evaluate the agent
max_timesteps_per_stage = 500000  # Maximum timesteps to train per stage to prevent infinite loops

#  Check for Existing Saved Models Before Training**
# Initialize the latest_stage variable
latest_stage = 0
for stage in range(0,len(curriculum)):
    print('#############')
    print(f'stage:{stage}')
    print('##################')
    PPO_Path = os.path.join('Training_008', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}.zip')
    if os.path.exists(PPO_Path):
        latest_stage = stage
        print(f"Found saved model for Stage {stage}. Loading the model.")
        # Generate the custom map for this stage
        hole_percentage = curriculum[stage - 1]
        custom_map = ut.generate_valid_map(size, hole_percentage)
        # Print the custom map
        ut.print_mapprint_map(custom_map)
        # Create the training environment
        train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
        train_env = Monitor(train_env)
        train_env = DummyVecEnv([lambda: train_env])
        # Load the model
        model = PPO.load(PPO_Path, env=train_env, device=device)
        break  # Exit after loading the latest model

if model is None:
    print("No saved model found. Starting from scratch.")
    latest_stage = 0  # Start from stage 1

# Loop through each stage in the curriculum
for stage, hole_percentage in enumerate(curriculum, 1):
    if stage <= latest_stage:
        print(f"Stage {stage} already trained. Skipping this stage.")
        continue  # Skip stages that have already been trained

    print(f"\nStage {stage}: Training with hole percentage {hole_percentage * 100}%")

    # Generate the custom map
    custom_map = ut.generate_valid_map(size, hole_percentage)

    # Print the custom map
    ut.print_map(custom_map)

    # Create the training environment without rendering
    train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])

    # Create the evaluation environment without rendering
    eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Initialize or update the model
    if model is None:
        # First stage: Initialize the model
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,  # Use the device determined earlier
            learning_rate=linear_schedule(3e-4)
        )
    else:
        # Subsequent stages: Update the environment
        model.set_env(train_env)

    # Custom callback to check performance and implement threshold
    class ThresholdCallback(BaseCallback):
        def __init__(self, eval_env, threshold, eval_episodes, verbose=0):
            super(ThresholdCallback, self).__init__(verbose)
            self.eval_env = eval_env
            self.threshold = threshold
            self.eval_episodes = eval_episodes
            self.success_rate = 0.0
            self.stage_timesteps = 0

        def _on_step(self) -> bool:
            self.stage_timesteps += self.locals['n_steps']
            # Evaluate every 5000 timesteps
            if self.stage_timesteps % 5000 == 0:
                mean_reward, _ = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.eval_episodes,
                    render=False,
                    deterministic=True,
                    warn=False
                )
                self.success_rate = mean_reward  # Since reward is 1 for success and 0 for failure
                print(f"Evaluation at timestep {self.num_timesteps}: Success Rate = {self.success_rate * 100}%")
                if self.success_rate >= self.threshold:
                    print(f"Success rate {self.success_rate * 100}% >= threshold {self.threshold * 100}%. Moving to next stage.")
                    return False  # Returning False stops training
            return True

    # Instantiate the threshold callback
    threshold_callback = ThresholdCallback(
        eval_env=eval_env,
        threshold=performance_threshold,
        eval_episodes=evaluation_episodes,
        verbose=1
    )

    # Combine the callbacks
    callback = CallbackList([threshold_callback])


    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=max_timesteps_per_stage, callback=threshold_callback)
    print(f"Training Completed for Stage {stage}")

    # Save the model
    PPO_Path = os.path.join('Training_008', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')
    os.makedirs(os.path.dirname(PPO_Path), exist_ok=True)  # Ensure the directory exists
    print('Saving the model')
    model.save(PPO_Path)

# ###################### Evaluating the model ###################
# **Modification 2: Update PPO_Path to the last saved model**
if latest_stage == len(curriculum):
    # If the last stage was already trained and loaded
    PPO_Path = os.path.join('Training_008', f'Saved_Models_Stage_{latest_stage}', f'FrozenLake_Stage_{latest_stage}')
else:
    # Use the last model trained in the loop
    PPO_Path = os.path.join('Training_008', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')

# Setting the env for evaluation with the hardest difficulty
final_hole_percentage = curriculum[-1]
print(f"\nEvaluating the final model on hole percentage {final_hole_percentage * 100}%")

# Generate the final evaluation map
final_map = ut.generate_valid_map(size, final_hole_percentage)
ut.print_map(final_map)

env = gym.make(environment_name, desc=final_map, is_slippery=True, render_mode='human')
env = Monitor(env)
eval_env = DummyVecEnv([lambda: env])

# Load the final model
model = PPO.load(PPO_Path, env=env, device=device)

# Evaluation
print('Evaluating the model...')
episode_rewards, episode_lengths = evaluate_policy(
    model, eval_env, n_eval_episodes=10, render=True, return_episode_rewards=True
)
print('Evaluation completed')

# Print the reward for each episode
for idx, reward in enumerate(episode_rewards):
    print(f"Episode {idx + 1}: Reward = {reward}")

env.close()
