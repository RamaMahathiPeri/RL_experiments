# IMPORT DEPENDENCIES
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from typing import Callable
import os
import torch

# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal) backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

# ENVIRONMENT
environment_name = 'CarRacing-v2'
env = gym.make(environment_name)

# TESTING THE ENVIRONMENT
obs, info = env.reset()

# Run episodes to test the environment
episodes = 1
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Check if the episode has ended
        score += reward

    print(f'Episode: {episode} Score: {score}')

env.close()

# TRAINING THE MODEL
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)  # Normalizing environment
env = VecFrameStack(env, n_stack=4)  # Frame stacking
env = VecMonitor(env)

log_path = os.path.join('Training', 'Logs')

# Define policy kwargs to disable image normalization
policy_kwargs = dict(normalize_images=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path, device="mps",
            learning_rate=linear_schedule(3e-4), policy_kwargs=policy_kwargs)

print('Training the model')

# Define the checkpoint callback to save the model every n steps
save_freq = 10000  # Save every 10000 steps
checkpoint_dir = os.path.join('Training', 'Saved Models', 'checkpoints_Cnn150000')
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=checkpoint_dir,
    name_prefix='Car_model_Cnn150000'
)

# Save the VecNormalize statistics during training
vec_normalize_stats_path = os.path.join('Training', 'Saved Models', 'vec_normalize.pkl')

# Combine the callbacks
callback = CallbackList([checkpoint_callback])

print('Ready to learn ********')

# Train the model
model.learn(total_timesteps=150000, callback=callback)

# Save VecNormalize statistics for later use
env.save(vec_normalize_stats_path)

print('Training Completed')

# Saving the model
PPO_Path = os.path.join('Training', 'Saved Models', 'Model_RacingCar_Cnn150000')
print('Saving the model')
model.save(PPO_Path)

# Re-loading the model
del model
model = PPO.load(PPO_Path, env=env, device="mps")

# Create the evaluation environment, ensuring it's wrapped identically to the training environment
eval_env = gym.make(environment_name)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_obs=10.0)  # Ensure normalization is applied to eval env
eval_env = VecFrameStack(eval_env, n_stack=4)  # Frame stacking
eval_env = VecMonitor(eval_env)

# Load the saved VecNormalize statistics
eval_env = VecNormalize.load(vec_normalize_stats_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False
eval_env = VecMonitor(eval_env)


# Evaluation
print('Evaluating the model')
eval = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print('Evaluation completed')

eval_env.close()  # Closing the evaluation environment

# Test the model
print('Testing the model')
episodes = 30  # trying to loop the env 30 times
for episode in range(1, episodes + 1):
    obs = env.reset()  # setting the initial observation
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)  # Predicting actions using the model
        obs, reward, done, info = env.step(action)
        score += reward
    print(f'Episode: {episode} Score: {score}')

env.close()

# Viewing the logs in Tensorboard
training_log_path = os.path.join('/Users/ramamahathiperi/Desktop/RL_Nick/02_Car-environment/Training/Logs/')
print(training_log_path)
# tensorboard --logdir=Logs (command to be used in terminal)
