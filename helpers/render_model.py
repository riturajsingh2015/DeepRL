import PIL
import time
import gym
import os
import numpy as np
import pandas as pd

try:
    import pyvirtualdisplay
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
    #print("pyvirtualdisplay successfully imported")
except ImportError:
    print("pyvirtualdisplay not imported")
    pass


def save_gif_N_get_path(frames=None):
    PROJECT_ROOT_DIR="."       
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,"IMAGES")   
       
    os.makedirs(IMAGES_PATH, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    IMAGES_PATH = os.path.join(IMAGES_PATH, "file_"+timestr+".gif")
    
    frames=frames
    frame_images = [PIL.Image.fromarray(frame) for frame in frames]
    frame_images[0].save(IMAGES_PATH, format='GIF',append_images=frame_images[1:],save_all=True,duration=20,loop=0)
    
    return IMAGES_PATH

def render_model(model,env_name=None, seed=42):
    frames = []
    env = gym.make(env_name)
    env.seed(seed)
    obs = env.reset()
    done = False
    reward_per_step=[]
    #ep_reward=0
    while not done:
        frames.append(env.render(mode="rgb_array"))        
        pred_action = np.argmax(model.predict(obs.reshape(1, -1)))
        obs, reward, done, info = env.step(pred_action)
        #ep_reward+=reward
        reward_per_step.append(reward)
        #reward_per_step.append(ep_reward)

    env.close()
    return save_gif_N_get_path(frames=frames) , reward_per_step, sum(reward_per_step)


def test_instances_of_env(test_cases=None,env_name=None,model=None,):
    image_paths=[]
    test_data = {
        'Tests': [],
        'Rewards': [],
        'Steps': []             
        }
    for case in test_cases:
        image_path,reward_per_step,total_reward= render_model(model,env_name=env_name, seed=case)
        image_paths.append(image_path)
        print("Tests # {} , Rewards {} , Steps {}".format(case, total_reward ,len(reward_per_step) ))
        test_data['Tests'].append(case)
        test_data['Rewards'].append(total_reward)
        test_data['Steps'].append(len(reward_per_step))
    
    #df = pd.DataFrame(test_data)
    
    df = pd.DataFrame({'Rewards': test_data['Rewards'],
                   'Steps': test_data['Steps']}, index=test_data['Tests'])
    df.loc['mean'] = df.mean()
    
    return df,image_paths

