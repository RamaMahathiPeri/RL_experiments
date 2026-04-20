import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Custom maps
level1_map = [
    'SFFF',
    'FFFF',
    'FFFF',
    'FFFF'
]

level2_map = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]

# Environment wrapper
class FrozenLakeEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FrozenLakeEnvWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=15, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return np.array([state], dtype=np.float32), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return np.array([state], dtype=np.float32), reward, terminated, truncated, info

# Curriculum levels
curriculum = [
    {'map': level1_map, 'is_slippery': False, 'timesteps': 5000},
    {'map': level2_map, 'is_slippery': False, 'timesteps': 10000},
    {'map': level2_map, 'is_slippery': True, 'timesteps': 15000},
]

policy_kwargs = dict(net_arch=[64, 64])

model = None

for idx, level in enumerate(curriculum):
    print(f"\nTraining on Level {idx + 1}")
    env = gym.make(
        'FrozenLake-v1',
        desc=level['map'],
        is_slippery=level['is_slippery']
    )
    env = FrozenLakeEnvWrapper(env)
    env = make_vec_env(lambda: env, n_envs=1)

    eval_env = gym.make(
        'FrozenLake-v1',
        desc=level['map'],
        is_slippery=level['is_slippery']
    )
    eval_env = FrozenLakeEnvWrapper(eval_env)
    eval_env = make_vec_env(lambda: eval_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./logs/level_{idx + 1}',
        log_path=f'./logs/level_{idx + 1}',
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    if model is None:
        model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    else:
        model.set_env(env)

    model.learn(total_timesteps=level['timesteps'], callback=eval_callback)

    # Save the model
    model.save(f'ppo_frozenlake_level_{idx + 1}')

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"Level {idx + 1} Mean Reward: {mean_reward}")

    # Check progression criteria
    if mean_reward < 0.8:
        print("Agent did not meet the progression criteria.")
        break
