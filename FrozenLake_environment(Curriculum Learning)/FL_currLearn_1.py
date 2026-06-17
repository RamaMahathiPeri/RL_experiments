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

# Function to create a custom map with random start and goal positions at the corners
def create_custom_map(size, hole_percentage):
    # Calculate the number of holes
    total_tiles = size * size
    num_holes = int(total_tiles * hole_percentage)
    num_frozen = total_tiles - num_holes - 2  # Exclude start and goal

    # Initialize the map with 'F' tiles
    lake_map = ['F'] * total_tiles

    # Define the corner positions in flat index
    corner_positions = [0, size - 1, (size - 1) * size, total_tiles - 1]

    # Randomly select start and goal positions from corners
    start_pos_flat, goal_pos_flat = np.random.choice(corner_positions, size=2, replace=False)

    # Set the start and goal positions
    lake_map[start_pos_flat] = 'S'
    lake_map[goal_pos_flat] = 'G'

    # Exclude start and goal from possible hole positions
    possible_hole_positions = [i for i in range(total_tiles) if i != start_pos_flat and i != goal_pos_flat]

    # Randomly select positions for holes
    np.random.seed(None)  # Use system randomness
    hole_positions = np.random.choice(possible_hole_positions, num_holes, replace=False)

    # Place holes on the map
    for pos in hole_positions:
        lake_map[pos] = 'H'

    # Convert the flat list into a 2D grid
    lake_grid = [lake_map[i * size:(i + 1) * size] for i in range(size)]
    return lake_grid

# Function to check if there's a valid path from 'S' to 'G'
def is_valid_path(lake_grid):
    size = len(lake_grid)
    visited = [[False for _ in range(size)] for _ in range(size)]
    stack = []

    # Find the starting position
    start_pos = None
    for x in range(size):
        for y in range(size):
            if lake_grid[x][y] == 'S':
                start_pos = (x, y)
                break
        if start_pos is not None:
            break

    if start_pos is None:
        raise Exception("Start position 'S' not found in the map.")

    stack.append(start_pos)

    while stack:
        x, y = stack.pop()
        if not (0 <= x < size and 0 <= y < size):
            continue
        if visited[x][y]:
            continue
        if lake_grid[x][y] == 'H':
            continue
        if lake_grid[x][y] == 'G':
            return True  # Path found

        visited[x][y] = True

        # Add neighboring positions
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < size and 0 <= ny < size:
                if not visited[nx][ny] and lake_grid[nx][ny] != 'H':
                    stack.append((nx, ny))

    return False  # No path found

# Function to generate a valid map
def generate_valid_map(size, hole_percentage):
    max_attempts = 1000
    for attempt in range(max_attempts):
        lake_grid = create_custom_map(size, hole_percentage)
        if is_valid_path(lake_grid):
            print(f"Valid map generated after {attempt + 1} attempts.")
            return lake_grid
    raise Exception("Failed to generate a valid map after 1000 attempts.")

# Function to print the custom map
def print_map(lake_grid):
    print("Custom Frozen Lake Map:")
    for row in lake_grid:
        print(' '.join(row))

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
unique_code= get_unique_code()


############################################################################################################

# Define the curriculum stages with increasing hole percentages
curriculum = [0.01, 0.05, 0.1]   # Start with 1% holes, then 5%, then 10%.

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# Define the path for logging
log_path = os.path.join(f'Training_{unique_code}', 'Logs_curriculum_threshold')

# Initialize the model variable (will be assigned during the first stage)
model = None

# Define the performance threshold based on mean reward
performance_threshold = 0.90  # Agent must achieve a mean reward of 0.90 to move to next stage
evaluation_episodes = 5       # Number of episodes to evaluate the agent
max_timesteps_per_stage = 500000  # Maximum timesteps to train per stage to prevent infinite loops

# Start the loop over stages
for stage, hole_percentage in enumerate(curriculum, 1):
    # Determine the path to the model for the current stage
    current_model_path = os.path.join(f'Training_{unique_code}', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}.zip')

    # Determine the path to the model for the previous stage
    if stage > 1:
        previous_model_path = os.path.join(f'Training_{unique_code}', f'Saved_Models_Stage_{stage - 1}', f'FrozenLake_Stage_{stage - 1}.zip')
    else:
        previous_model_path = None

    # Check if the model for the current stage already exists
    if os.path.exists(current_model_path):
        print(f"Model for Stage {stage} already exists. Loading the model.")
        # Generate the custom map for this stage
        custom_map = generate_valid_map(size, hole_percentage)
        # Print the custom map
        print_map(custom_map)
        # Create the training environment
        train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
        train_env = Monitor(train_env)
        train_env = DummyVecEnv([lambda: train_env])
        # Load the model
        model = PPO.load(current_model_path, env=train_env, device=device)
        continue  # Skip training for this stage
    else:
        # If model for current stage doesn't exist
        # Check if we can load the model from the previous stage
        if previous_model_path and os.path.exists(previous_model_path):
            print(f"Loading model from Stage {stage - 1} to initialize training for Stage {stage}.")
            # Generate the custom map for the previous stage
            hole_percentage_prev = curriculum[stage - 2]
            custom_map_prev = generate_valid_map(size, hole_percentage_prev)
            # Create the training environment for the previous stage
            train_env_prev = gym.make(environment_name, desc=custom_map_prev, is_slippery=True, render_mode=None)
            train_env_prev = Monitor(train_env_prev)
            train_env_prev = DummyVecEnv([lambda: train_env_prev])
            # Load the model from the previous stage
            model = PPO.load(previous_model_path, env=train_env_prev, device=device)
            # Now update the environment to the current stage
            custom_map = generate_valid_map(size, hole_percentage)
            print_map(custom_map)
            train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
            train_env = Monitor(train_env)
            train_env = DummyVecEnv([lambda: train_env])
            model.set_env(train_env)
        else:
            # If no model from previous stage, initialize a new model
            print(f"No model from previous stage. Initializing new model for Stage {stage}.")
            # Generate the custom map
            custom_map = generate_valid_map(size, hole_percentage)
            print_map(custom_map)
            train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
            train_env = Monitor(train_env)
            train_env = DummyVecEnv([lambda: train_env])
            model = PPO(
                'MlpPolicy',
                train_env,
                verbose=1,
                tensorboard_log=log_path,
                device=device,
                learning_rate=linear_schedule(3e-4)
            )

    # Create the evaluation environment
    eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

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
                self.mean_reward = mean_reward
                print(f"Evaluation at timestep {self.num_timesteps}: Mean Reward = {self.mean_reward}")
                if self.mean_reward >= self.threshold:
                    print(f"Mean reward {self.mean_reward} >= threshold {self.threshold}. Moving to next stage.")
                    return False  # Returning False stops training
            return True

    # Instantiate the threshold callback
    threshold_callback = ThresholdCallback(
        eval_env=eval_env,
        threshold=performance_threshold,
        eval_episodes=evaluation_episodes,
        verbose=1
    )

    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=max_timesteps_per_stage, callback=threshold_callback)
    print(f"Training Completed for Stage {stage}")

    # Save the model
    PPO_Path = os.path.join(f'Training_{unique_code}', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')
    os.makedirs(os.path.dirname(PPO_Path), exist_ok=True)  # Ensure the directory exists
    print('Saving the model')
    model.save(PPO_Path)

# ###################### Evaluating the model ###################
# Update PPO_Path to the last saved model
PPO_Path = os.path.join(f'Training_{unique_code}', f'Saved_Models_Stage_{len(curriculum)}', f'FrozenLake_Stage_{len(curriculum)}')

# Setting the env for evaluation with the hardest difficulty
final_hole_percentage = curriculum[-1]
print(f"\nEvaluating the final model on hole percentage {final_hole_percentage * 100}%")

# Generate the final evaluation map    
final_map = generate_valid_map(size, final_hole_percentage)
print_map(final_map)

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
