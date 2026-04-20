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
env = gym.make(environment_name,desc=None,map_name='8x8',is_slippery=True)
# Wrap the environment with Monitor to record episode stats
env = Monitor(env)
# Use DummyVecEnv and VecMonitor for vectorized environment and logging
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

DQN_Path ='/Users/ramamahathiperi/Desktop/RL_Nick/03_FrozenLake-environment /Training/Saved Models_3l_8x8/FrozenLake_Dqn3l.zip'



model = DQN.load(DQN_Path, env=env, device=device)

# Create the evaluation environment with rendering for evaluation
eval_env = gym.make(environment_name, render_mode='human',desc=None,map_name='8x8',is_slippery=True)
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecMonitor(eval_env)

# Evaluation
print('Evaluating the model')
episode_rewards, episode_lengths = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    render=True,
    return_episode_rewards=True
)
print('Evaluation completed')

# Print the reward for each episode
for idx, reward in enumerate(episode_rewards):
    print(f"Episode {idx + 1}: Reward = {reward}")

eval_env.close()  # Closing the evaluation environment


# Viewing the logs in Tensorboard
# Run the following command in your terminal:
# tensorboard --logdir=Training/Logs_2l/
