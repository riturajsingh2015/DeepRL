import PIL
import time
import gym
import os, shutil
import numpy as np
import pandas as pd



class Rendering:
    def __init__(self,
                 env_name="CartPole-v1",
                 test_model=None,
                 case_list=[42,900,930,180,660,240,960,450,30,90,150,210,330,420,510,75],
                 dir_path=None
                ):  
        try:
            import pyvirtualdisplay
            _ = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
            print("\rpyvirtualdisplay successfully imported", end="")
        except ImportError:
            print("\rpyvirtualdisplay not imported", end="")
            pass
        self.env_name=env_name
        self.model=test_model
        self.test_cases=case_list
        self.dir_path=dir_path



    def save_gif_N_get_path(self,frames=None,seed_val=None):        
        IMAGES_PATH = os.path.join(self.dir_path, "IMAGES")
        
        os.makedirs(IMAGES_PATH, exist_ok=True)
        
        IMAGES_PATH = os.path.join(IMAGES_PATH, "img_"+str(seed_val)+".gif")
        
        frame_images = [PIL.Image.fromarray(frame) for frame in frames]
        frame_images[0].save(IMAGES_PATH, format='GIF',append_images=frame_images[1:],save_all=True,duration=20,loop=0)
        
        return IMAGES_PATH

    def render_model(self,seed=None):
        frames = []
        env = gym.make(self.env_name)
        env.seed(seed)
        obs = env.reset()
        done = False
        
        ep_reward = 0
        ep_steps=0
        
        while not done:
            frames.append(env.render(mode="rgb_array"))        
            pred_action = np.argmax(self.model.predict(obs.reshape(1, -1)))
            obs, reward, done, info = env.step(pred_action)
            ep_reward += reward
            ep_steps+=1          

        env.close()
        return self.save_gif_N_get_path(frames=frames,seed_val=seed) , ep_steps, ep_reward


    def test_instances_of_env(self):
        image_paths=[]
        test_data = {
            'Case#': [],
            'Rewards': [],
            'Steps': []             
            }
        for case in self.test_cases:
            image_path,ep_steps, ep_reward= self.render_model(seed=case)
            print("\rCase# {} , Steps {} , Total rewards {}".format(case,ep_steps,ep_reward), end="")
            image_paths.append(image_path)
            test_data['Case#'].append(case)
            test_data['Rewards'].append(ep_reward)
            test_data['Steps'].append(ep_steps)

        df = pd.DataFrame({'Rewards': test_data['Rewards'],
                           'Steps': test_data['Steps']}, 
                          index=test_data['Case#'])
        df.loc['mean'] = df.mean()
        return df,image_paths

