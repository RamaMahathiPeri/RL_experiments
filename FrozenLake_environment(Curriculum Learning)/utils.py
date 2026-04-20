import numpy as np
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

    