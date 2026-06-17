#IMPORT DEPENDENCIES

import gymnasium as gym
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor,VecNormalize,VecFrameStack,VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback,CallbackList
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
import numpy as np 


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
DQN_Path = ('/Users/ramamahathiperi/Desktop/RL_Nick/03_FrozenLake-environment /Training/Saved Models_custom_5l/FrozenLake_custom_5l.zip')

# Function to create a custom map
def create_custom_map(size=10, hole_percentage=0.5):
    # Calculate the number of holes
    total_tiles = size * size
    num_holes = int(total_tiles * hole_percentage)
    num_frozen = total_tiles - num_holes - 2  # Exclude start and goal

    # Initialize the map with 'F' tiles
    lake_map = ['F'] * total_tiles

    # Set the start and goal positions
    lake_map[0] = 'S'             # Start at top-left corner
    lake_map[-1] = 'G'            # Goal at bottom-right corner

    # Exclude start and goal from possible hole positions
    possible_hole_positions = list(range(1, total_tiles - 1))

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
    start_pos = (0, 0)

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
        neighbors = [ (x-1, y), (x+1, y), (x, y-1), (x, y+1) ]
        for nx, ny in neighbors:
            if 0 <= nx < size and 0 <= ny < size:
                if not visited[nx][ny] and lake_grid[nx][ny] != 'H':
                    stack.append((nx, ny))

    return False  # No path found

# Function to generate a valid map
def generate_valid_map(size=10, hole_percentage=0.5):
    max_attempts = 1000
    for attempt in range(max_attempts):
        lake_grid = create_custom_map(size, hole_percentage)
        if is_valid_path(lake_grid):
            print(f"Valid map generated after {attempt + 1} attempts.")
            return lake_grid
    raise Exception("Failed to generate a valid map after 1000 attempts.")

# Generate the custom map
custom_map = generate_valid_map(size=10, hole_percentage=0.5)

# Function to print the custom map
def print_map(lake_grid):
    print("Custom Frozen Lake Map:")
    for row in lake_grid:
        print(' '.join(row))

print_map(custom_map)
# Testing the custom environment 

env = gym.make(environment_name, desc=custom_map, is_slippery=True,render_mode='human')
env = Monitor(env)
eval = DummyVecEnv([lambda:env])


model = DQN.load(DQN_Path,env=env,device="mps")








# Evaluation
print('Evaluating the model...')
episode_rewards, episode_lengths= evaluate_policy(model, env, n_eval_episodes=10, render=True,return_episode_rewards=True)
print('Evaluation completed')

# Print the reward for each episode
for idx, reward in enumerate(episode_rewards):
    print(f"Episode {idx + 1}: Reward = {reward}")

env.close()
