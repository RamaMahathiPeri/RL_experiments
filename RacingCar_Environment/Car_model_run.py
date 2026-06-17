#IMPORT DEPENDENCIES 
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
import os 
import pygame
pygame.init()
import torch
from stable_baselines3.common.monitor import Monitor


#ENVIRONMENT 
environment_name = 'CarRacing-v2'
env = gym.make(environment_name,render_mode='human')
env = Monitor(env)

env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)


PPO_Path = '/Users/ramamahathiperi/Desktop/RL_Nick/02_Car-environment /Training/Saved Models/PPO_Model_RacingCar'

model = PPO.load(PPO_Path,env=env,device="mps")

# Evaluation

eval = evaluate_policy(model, env, n_eval_episodes= 30,render=True)
print(eval)

env.close # closing the environment 

# Test the model 

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
training_log_path= os.path.join('/Users/ramamahathiperi/Desktop/RL_Nick/02_Car-environment /Training/Logs/','PPO_2')
print(training_log_path)
# tensorboard --logdir=(training_log_path) (command to be used in terminal )
