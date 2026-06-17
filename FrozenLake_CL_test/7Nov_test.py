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

print(os.getcwd())

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
log_path = os.path.join('Training', 'Logs_curriculum_threshold')

# Initialize the model variable (will be assigned during the first stage)
model = None

# Define the performance threshold
performance_threshold = 0.90  # Agent must achieve 90% success rate to move to next stage
evaluation_episodes = 1000     # Number of episodes to evaluate the agent
max_timesteps_per_stage = 500000  # Maximum timesteps to train per stage to prevent infinite loops



for stage,hole_percentage in enumerate(curriculum,1):
    if stage == 1:
        print('we are in stage 1: starting a new model')
        PPO_Path= os.path.join('training_flt_3',f'saved_models_stage_{stage}',f'frozenlake_stage_{stage}.zip')
        hole_percentage=curriculum[stage]
        custom_map= ut.generate_valid_map(size,hole_percentage)
        #print the custom map 
        print(f'printing the custom map for stage {stage}')
        ut.print_map(custom_map)
        #creating the training env 
        train_env=gym.make(environment_name,desc= custom_map,is_slippery=True,render_mode=None)
        train_env=Monitor(train_env)
        train_env= DummyVecEnv([lambda:train_env])
        #load the model
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            device=device,  # Use the device determined earlier
            learning_rate=linear_schedule(3e-4)
        )
    else:
        print(f'we are in stage{stage}')
        PPO_Path= os.path.join('training_flt_3',f'saved_models_stage_{stage-1}',f'frozenlake_stage_{stage-1}.zip')
        print('######')
        print(PPO_Path)
        print('########')
        hole_percentage=curriculum[stage-1]
        custom_map= ut.generate_valid_map(size,hole_percentage)
        #print the custom map 
        print(f'printing the custom map for stage {stage}')
        ut.print_map(custom_map)
        #creating the training env 
        train_env=gym.make(environment_name,desc= custom_map,is_slippery=True,render_mode=None)
        train_env=Monitor(train_env)
        train_env= DummyVecEnv([lambda:train_env])
        #load the model
        model= PPO.load(PPO_Path,env=train_env,device=device)
    
    # Create the evaluation environment without rendering
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
            self.success_rate = 0.0
            self.stage_timesteps = 0

        def _on_step(self) -> bool:
            self.stage_timesteps += self.locals['n_steps']
            # Evaluate every 10000 timesteps
            if self.stage_timesteps % 10000 == 0:
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
    PPO_Path = os.path.join('training_flt_3', f'Saved_Models_Stage_{stage}', f'FrozenLake_Stage_{stage}')
    print('Saving the model')
    model.save(PPO_Path)

    


