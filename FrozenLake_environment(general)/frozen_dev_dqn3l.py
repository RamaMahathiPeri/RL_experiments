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

# Create the environment with render_mode='human' for testing purposes
test_env = gym.make(environment_name, desc=None, map_name='8x8', is_slippery=True, render_mode='human')

# TESTING THE ENVIRONMENT
obs, info = test_env.reset()

# Run episodes to test the environment
episodes = 2
for episode in range(1, episodes + 1):
    obs, info = test_env.reset()
    done = False
    score = 0

    while not done:
        action = test_env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated  # Check if the episode has ended
        score += reward

    print(f'Episode: {episode} Score: {score}')

test_env.close()

# TRAINING THE MODEL

# Create the training environment
env = gym.make(environment_name)
# Wrap the environment with Monitor to record episode stats
env = Monitor(env)
# Use DummyVecEnv and VecMonitor for vectorized environment and logging
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

# Define the path for logging
log_path = os.path.join('Training', 'Logs_3l')

# Define a linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# Initialize the model
model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    tensorboard_log=log_path,
    device=device,  # Use the device determined earlier
    learning_rate=linear_schedule(3e-4)
)

print('Training the model')

# Define the checkpoint callback to save the model every n steps
save_freq = 10000  # Save every 10,000 steps (adjust as needed)
checkpoint_dir = os.path.join('Training', 'Saved Models_3l', 'checkpoints_3l')
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=checkpoint_dir,
    name_prefix='FrozenLake_dqn3l'
)

# Create an evaluation environment
eval_env = gym.make(environment_name)
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecMonitor(eval_env)

# Define the evaluation callback to log evaluation rewards
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join('Training', 'Saved Models_3l', 'best_model'),
    log_path=os.path.join('Training', 'Logs_3l', 'eval'),
    eval_freq=5000,  # Evaluate every 5,000 steps
    deterministic=True,
    render=False
)

# Combine the callbacks
callback = CallbackList([checkpoint_callback, eval_callback])

print('Ready to learn ********')

# Train the model
model.learn(total_timesteps=300000, callback=callback)

print('Training Completed')

# Saving the model
DQN_Path = os.path.join('Training', 'Saved Models_3l', 'FrozenLake_Dqn3l')
print('Saving the model')
model.save(DQN_Path)

# Re-loading the model
del model
model = DQN.load(DQN_Path, env=env, device=device)

# Create the evaluation environment with rendering for evaluation
eval_env = gym.make(environment_name, render_mode='human')
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

# Test the model
print('Testing the model')
episodes = 5  # Adjust the number of test episodes as needed
for episode in range(1, episodes + 1):
    obs, info = eval_env.reset()
    done = False
    score = 0

    while not done:
        # Note: env.render() is not needed since render_mode='human' is set
        action, _ = model.predict(obs, deterministic=True)  # Predicting actions using the model
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        score += reward

    print(f'Episode: {episode} Score: {score}')

eval_env.close()

# Viewing the logs in Tensorboard
# Run the following command in your terminal:
# tensorboard --logdir=Training/Logs_2l/
