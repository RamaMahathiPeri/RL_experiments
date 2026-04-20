# IMPORT DEPENDENCIES
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import torch
import numpy as np
import utils as ut  
from typing import Callable

# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal) backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

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
log_path = os.path.join(f'Training_18Nov_{unique_code}', 'Logs_curriculum_threshold')

# Custom Environment Wrapper to change the map periodically
class ChangingMapFrozenLakeEnv(gym.Wrapper):
    def __init__(self, env, size=6, hole_percentage=0.05, is_slippery=True, change_interval=100):
        super(ChangingMapFrozenLakeEnv, self).__init__(env)
        self.size = size
        self.hole_percentage = hole_percentage
        self.is_slippery = is_slippery
        self.change_interval = change_interval
        self.step_count = 0

    def reset(self, **kwargs):
        # Reset the step count
        self.step_count = 0
        # Generate a new valid map
        custom_map = ut.generate_valid_map(self.size, self.hole_percentage)
        # Update the environment's map
        self.env = FrozenLakeEnv(desc=custom_map, is_slippery=self.is_slippery)
        observation, info = self.env.reset(**kwargs)  # Returns (observation, info)
        return observation, info

    def step(self, action):
        self.step_count += 1
        if self.step_count >= self.change_interval:
            # Reset the environment with a new map
            observation, info = self.reset()
            reward = 0
            terminated = True
            truncated = False
            # Info can be updated or kept as is
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

# Custom Observation Wrapper to convert scalar observations to grid
class GridObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_size):
        super(GridObservationWrapper, self).__init__(env)
        self.grid_size = grid_size
        # Update the observation space to a 3D Box (channels, height, width)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1, grid_size, grid_size),
            dtype=np.float32
        )

    def observation(self, observation):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Map the scalar observation to grid coordinates
        row = observation // self.grid_size
        col = observation % self.grid_size
        grid[row, col] = 1.0  # Mark the agent's position
        # Add a channel dimension
        return grid[np.newaxis, :, :]

# Custom CNN Features Extractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# Policy keyword arguments with the custom CNN
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128)
)


# Training loop for curriculum stages
for stage, hole_percentage in enumerate(curriculum, 1):
    print(f"\n=== Stage {stage}: Hole Percentage {hole_percentage*100}% ===")
    PPO_Path = os.path.join(f'Training_18Nov_{unique_code}', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')

    # Create the base environment
    base_env = gym.make(environment_name, is_slippery=True, render_mode='human')

    # Wrap the environment to change the map periodically
    train_env = ChangingMapFrozenLakeEnv(
        base_env,
        size=size,
        hole_percentage=hole_percentage,
        is_slippery=True,  
        change_interval=100  
    )
    # Wrap with GridObservationWrapper
    train_env = GridObservationWrapper(train_env, size)
    # Ensure wrappers are applied in the correct order
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])

    test_env=ut.testEnv(train_env)

    # Create the evaluation environment
    eval_env_base = gym.make(environment_name, is_slippery=True, render_mode=None)
    eval_env = ChangingMapFrozenLakeEnv(
        eval_env_base,
        size=size,
        hole_percentage=hole_percentage,
        is_slippery=True,
        change_interval=100
    )
    eval_env = GridObservationWrapper(eval_env, size)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Initialize or load the model
    if stage == 1:
        print('Initializing a new model with CNN.')
        model = PPO(
            'CnnPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
            learning_rate=linear_schedule(3e-4),  # Increased learning rate to help adapt
            n_steps=4096,  # Increased n_steps for more experience per update
            batch_size=256,  # Adjusted batch size
            gamma=0.98,  # Slightly lower gamma to focus on immediate rewards
            gae_lambda=0.9,  # Lower gae_lambda to reduce variance
            policy_kwargs=policy_kwargs
        )
    else:
        prev_PPO_Path = os.path.join(f'Training_18Nov_{unique_code}',f'saved_models_stage_{stage-1}',f'frozenlake_stage_{stage-1}.zip')
        print(f'Loading model from {prev_PPO_Path}')
        # Load the model from the previous stage
        model = PPO.load(prev_PPO_Path, env=train_env, device=device)

    # Define the callback to stop training when average reward reaches 0.8 (lowered due to increased difficulty)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.8, verbose=1)

    # Define the EvalCallback
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=10000,  # Increased eval_freq
                                 n_eval_episodes=20,  # Increased number of evaluation episodes
                                 best_model_save_path=os.path.join(PPO_Path, 'best_model'),
                                 verbose=1)

    # Train the model
    print(f"Training the model for Stage {stage}...")
    model.learn(total_timesteps=10_000_000, callback=eval_callback)  
    print(f"Training Completed for Stage {stage}")

    # Save the model
    os.makedirs(os.path.dirname(PPO_Path), exist_ok=True)  # Ensure the directory exists
    print('Saving the model')
    model.save(PPO_Path)

    # Load the best model saved during training for evaluation
    best_model_path = os.path.join(PPO_Path, 'best_model', 'best_model.zip')
    if os.path.exists(best_model_path):
        print(f'Loading best model from {best_model_path}')
        model = PPO.load(best_model_path, env=eval_env, device=device)
    else:
        print('Best model not found, using the last trained model.')

    # Evaluation
    print('Evaluating the model...')
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=20, render=True, return_episode_rewards=True)
    print('Evaluation completed')

    # Print the reward for each episode
    for idx, reward in enumerate(episode_rewards):
        print(f"Episode {idx + 1}: Reward = {reward}")

    eval_env.close()
