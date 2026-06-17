#IMPORT DEPENDENCIES

import gymnasium as gym
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor,VecNormalize,VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback,CallbackList
from stable_baselines3.common.utils import get_linear_fn

from typing import Callable
import os 
import pygame
#pygame.init()
import torch


#from gymnasium.wrappers.monitoring import video_recorder
#from gymnasium.wrappers.record_video import RecordVideo



# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal) backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

#ENVIRONMENT 
environment_name = 'CarRacing-v2'
env = gym.make(environment_name)

# TESTING THE ENVIRONMENT 

# Reset the environment (handle both observation and info)
obs, info = env.reset()

# Run episodes to test the environment
episodes = 1
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        #env.render()  # Render each frame
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Check if the episode has ended
        score += reward
    
    print(f'Episode: {episode} Score: {score}')

env.close()



# TRAINING THE MODEL 

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)  # Frame stacking
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)# Helps to keep a track of the rewards of the 
                                                                                    #stats across the environments 
#env = VecFrameStack(env, n_stack=4)  # Frame stacking
env = VecMonitor(env)

log_path = os.path.join('Training','Logs')

# Define policy kwargs to disable image normalization
policy_kwargs = dict(normalize_images=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path,device="mps",learning_rate=linear_schedule(0.001),policy_kwargs=policy_kwargs)
#model = PPO('CnnPolicy',env,verbose=1,tensorboard_log=log_path)
print('Training the model')

# Setup callbacks for saving models and evaluating policy
#checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./checkpoints/', name_prefix='ppo_cartpole')
#eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=100000, n_eval_episodes=5)

# Define the checkpoint callback to save the model every n steps
save_freq = 10000  # Save every 10000 steps
checkpoint_dir = os.path.join('Training', 'Saved Models', 'checkpoints_4m')
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=checkpoint_dir,
    name_prefix='Car_model_4m'
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=os.path.join('Training', 'Saved Models', 'best_model_4m'),
    log_path=os.path.join('Training', 'Logs', 'eval_results_4m'),
    eval_freq=1000000,  # Evaluate every 1,000,000 steps
    deterministic=True,
    render=False
)

# Combine the callbacks
callback = CallbackList([checkpoint_callback, eval_callback])

print ('ready to learn ********')


model.learn(total_timesteps=4000000,callback=callback)

print('Training Completed')
# saving the model 

PPO_Path = os.path.join('Training','Saved Models','Model_RacingCar_4m')

print('Saving the model')

print(PPO_Path)

model.save(PPO_Path)

# remodeling the model 
del model   

model = PPO.load(PPO_Path,env=env,device="mps")

# Evaluation
print('evaluating the model')
eval = evaluate_policy(model, env, n_eval_episodes= 10,render=True)
#print(eval)
print('evaluating completed')

env.close # closing the environment 

# Test the model 
print('Testing the model')
episodes = 30 # trying to loop the env 10 times 
for episode in range(1,episodes+1):
    print(env.reset())
    obs = env.reset() # setting the initial set of observation as intial state 
    done= False  # setting temp variables
    score= 0  # setting temp variables
    
    while not done:
        #env.render() # allows to view the graphical representation of the env 
        action,_ = model.predict(obs) # we are using the model here 
        print(action.shape)
        print('***********')
        print(env.step(action))
        obs,reward,done,info = env.step(action)
        score = score + reward
    print('Episode:{} Score:{}'.format(episode,score))
env.close()




##### viewing the logs in Tensorboard 
import os
training_log_path= os.path.join('/Users/ramamahathiperi/Desktop/RL_Nick/02_Car-environment /Training/Logs/')
print(training_log_path)
# tensorboard --logdir=Logs (command to be used in terminal )
