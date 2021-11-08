import PIL
import time
import gym
import os
import numpy as np
import pandas as pd



class Rendering:
    def __init__(self,
                 env_name="CartPole-v1",
                 model=None,
                 case_list=[42,900,930,180,660,240,960,450,30,90,150,210,330,420,510,75],
                 timestr=None
                ):  
      try:
          import pyvirtualdisplay
          _ = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
          print("pyvirtualdisplay successfully imported")
      except ImportError:
          print("pyvirtualdisplay not imported")
          pass
      self.env_name=env_name
      self.model=model
      self.test_cases=case_list
      self.timestr=timestr




    def save_gif_N_get_path(self,frames=None,seed_val=None):
        dir_path = os.path.join("DQN_trained_models", self.env_name ,"model_"+self.timestr)
        IMAGES_PATH = os.path.join(dir_path, "IMAGES")

        os.makedirs(IMAGES_PATH, exist_ok=True)
        #timestr=time.strftime("%Y%m%d-%H%M%S")
        IMAGES_PATH = os.path.join(IMAGES_PATH, "img_"+str(seed_val)+".gif")
        
        frames=frames
        frame_images = [PIL.Image.fromarray(frame) for frame in frames]
        frame_images[0].save(IMAGES_PATH, format='GIF',append_images=frame_images[1:],save_all=True,duration=20,loop=0)
        
        return IMAGES_PATH

    def render_model(self,seed=None):
        frames = []
        env = gym.make(self.env_name)
        env.seed(seed)
        obs = env.reset()
        done = False
        reward_per_step=[]
        
        while not done:
            frames.append(env.render(mode="rgb_array"))        
            pred_action = np.argmax(self.model.predict(obs.reshape(1, -1)))
            obs, reward, done, info = env.step(pred_action)
            reward_per_step.append(reward)
            

        env.close()
        return self.save_gif_N_get_path(frames=frames,seed_val=seed) , reward_per_step, sum(reward_per_step)


    def test_instances_of_env(self):
        image_paths=[]
        test_data = {
            'Tests': [],
            'Rewards': [],
            'Steps': []             
            }
        for case in self.test_cases:
            image_path,reward_per_step,total_reward= self.render_model(seed=case)
            image_paths.append(image_path)
            print("Tests # {} , Rewards {} , Steps {}".format(case, total_reward ,len(reward_per_step) ))
            test_data['Tests'].append(case)
            test_data['Rewards'].append(total_reward)
            test_data['Steps'].append(len(reward_per_step))

        df = pd.DataFrame({'Rewards': test_data['Rewards'],
                      'Steps': test_data['Steps']}, index=test_data['Tests'])
        df.loc['mean'] = df.mean()
        #print(df,image_paths)
        return df,image_paths

