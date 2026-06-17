# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
from typing import Callable
import os
import torch
import numpy as np 

# Set the working directory
os.chdir('/Users/ramamahathiperi/Desktop/RL_Nick/03_FrozenLake-environment ')
print(f"Current working directory: {os.getcwd()}")

# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal) backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

# ENVIRONMENT
environment_name = 'FrozenLake-v1'

""" # Create the environment with render_mode='human' for testing purposes
test_env = gym.make(environment_name, desc=None, map_name='8x8', is_slippery=True, render_mode='human')

# TESTING THE ENVIRONMENT
obs, info = test_env.reset()

# Run episodes to test the environment
episodes = 5
for episode in range(1, episodes + 1):
    obs, info = test_env.reset()
    print(obs,info)
    done = False
    score = 0

    while not done:
        action = test_env.action_space.sample()  # Take a random action
        print(action)
        print(test_env.step(action))
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated  # Check if the episode has ended
        score += reward

    print(f'Episode: {episode} Score: {score}')

test_env.close()
 """
# Function to create a custom map
def create_custom_map(size=10, hole_percentage=0.01):
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
custom_map = generate_valid_map(size=10, hole_percentage=0.01) 

# Function to print the custom map
def print_map(lake_grid):
    print("Custom Frozen Lake Map:")
    for row in lake_grid:
        print(' '.join(row))

print_map(custom_map)
# Testing the custom environment 

env = gym.make(environment_name, desc=custom_map, is_slippery=True,render_mode='human')


# TESTING THE ENVIRONMENT
obs, info =env.reset()
print(obs,info)

# Run episodes to test the environment
episodes = 5
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Check if the episode has ended
        score += reward

    print(f'Episode: {episode} Score: {score}')

env.close()
#################################################################
# TRAINING THE MODEL

# Create the training environment without rendering
train_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)

# Wrap the environment with Monitor to record episode stats
train_env = Monitor(train_env)

# Use DummyVecEnv for vectorized environment
train_env = DummyVecEnv([lambda: train_env])

# Define the path for logging
log_path = os.path.join('Training', 'Logs_custom_5l')

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# Initialize the model
model = DQN(
    'MlpPolicy',
    train_env,
    verbose=1,
    tensorboard_log=log_path,
    device=device,  # Use the device determined earlier
    learning_rate=linear_schedule(3e-4)
)

print('Training the model')

# Define the checkpoint callback to save the model every n steps
save_freq = 10000  # Save every 10,000 steps (adjust as needed)
checkpoint_dir = os.path.join('Training', 'Saved Models_custom_5l', 'checkpoints')
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=checkpoint_dir,
    name_prefix='FrozenLake_custom_dqn5l'
)

# Create an evaluation environment without rendering
eval_env = gym.make(environment_name, desc=custom_map, is_slippery=True, render_mode=None)
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

# Define the evaluation callback to log evaluation rewards
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join('Training', 'Saved Models_custom_5l', 'best_model'),
    log_path=os.path.join('Training', 'Logs_custom_5l', 'eval'),
    eval_freq=5000,  # Evaluate every 5,000 steps
    deterministic=True,
    render=False
)

# Combine the callbacks
callback = CallbackList([checkpoint_callback, eval_callback])

print('Ready to learn ********')

# Train the model
model.learn(total_timesteps=500000, callback=callback)

print('Training Completed')

# Saving the model
DQN_Path = os.path.join('Training', 'Saved Models_custom_5l', 'FrozenLake_custom_5l')
print('Saving the model')
model.save(DQN_Path)
