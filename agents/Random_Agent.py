import PIL
import time
import gym
import numpy as np
import pandas as pd
from datetime import datetime
import os, shutil


class Random_Agent(object):
    def __init__(self,env_name="CartPole-v1"):
        try:
            import pyvirtualdisplay
            _ = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
            print("\rpyvirtualdisplay successfully imported", end="")
        except ImportError:
            print("\rpyvirtualdisplay not imported", end="")
            pass
        self.env_name = env_name        
        self.env = gym.make(self.env_name)
        
    
    def play_an_episode(self):  
        frames = []
        ep_reward = 0
        ep_steps=0
        
        state = self.env.reset()
        done=False        
        while not done:
            frames.append(self.env.render(mode="rgb_array"))    
            # sample a random action
            action = self.env.action_space.sample()                
            # take that chosen action on envrionment
            state, reward, done, info = self.env.step(action)
            ep_reward += reward
            ep_steps+=1  
        self.env.close()
        return self.save_gif_N_get_path(frames=frames) , ep_steps, ep_reward

    def save_gif_N_get_path(self,frames=None):  
        dir_path = os.path.join("..","Random_models", self.env_name)
        os.makedirs(dir_path, exist_ok=True)
        IMAGES_PATH = os.path.join(dir_path, "IMAGES")        
        os.makedirs(IMAGES_PATH, exist_ok=True)
        
        now = datetime.now() # current date and time        
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        
        IMAGES_PATH = os.path.join(IMAGES_PATH, "img_"+date_time+".gif")
        
        frame_images = [PIL.Image.fromarray(frame) for frame in frames]
        frame_images[0].save(IMAGES_PATH, format='GIF',append_images=frame_images[1:],save_all=True,duration=20,loop=0)
        
        return IMAGES_PATH
        

        
