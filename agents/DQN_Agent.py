import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import os
import time
import pandas as pd
from helpers.render_model import *
from helpers.plot_util import *


'''
*****Notations used *********
Learning rate - alpha (α), 
Discount factor - gamma (γ), 
Batch-size - B, 
Maximum Capacity of ReplayMemory- C 
ReplayMemory - D
Q-function or Q-Network = Q
Number of Hidden layers in the Q-Network = 2
Number of neurons in layer 1 = layer1_size
Number of neurons in layer 2 = layer2_size

S=current state
S_new=new state
A=action
R=reward


'''

# ReplyMemory (C) is used to save the experience tuples
class ReplayMemory():
   
    def __init__(self, max_capacity, state_space):
        self.C = max_capacity

        self.C_counter = 0
        # Initialize a dict to save all the experience replated information 
        self.Memory_dict = {
        'states': np.zeros((self.C, state_space),dtype=np.float32),
        'new_states': np.zeros((self.C, state_space),dtype=np.float32),
        'actions': np.zeros(self.C, dtype=np.int32),
        'rewards': np.zeros(self.C, dtype=np.float32),
        'terminals': np.zeros(self.C, dtype=np.int32)
             
        }
    # This functions will save the transitions in the Memory    
    def save_transition(self, S, A, R, S_new, done):
        index = self.C_counter % self.C
        self.Memory_dict['states'][index] = S
        self.Memory_dict['new_states'][index] = S_new
        self.Memory_dict['rewards'][index] = R
        self.Memory_dict['actions'][index] = A
        self.Memory_dict['terminals'][index] = 1 - int(done)
        self.C_counter += 1

    # This functions samples a mini-batch of size B from the Memory
    def sample_mini_batch(self, B):
        max_mem = min(self.C_counter, self.C)
        batch = np.random.choice(max_mem, B, replace=False)

        S_js = self.Memory_dict['states'][batch]
        S_nexts = self.Memory_dict['new_states'][batch]
        R_js = self.Memory_dict['rewards'][batch]
        A_js = self.Memory_dict['actions'][batch]
        terminal = self.Memory_dict['terminals'][batch]

        return S_js, A_js,  R_js,  S_nexts, terminal


# DQN_Agent class used to create DQN agent related functionalities
class DQN_Agent():
    def __init__(self, 
                 alpha=0.001, 
                 env_name="CartPole-v1",                 
                 gamma=0.98, 
                 epsilon=1.0, 
                 B=64,
                 epsilon_dec=1e-3, 
                 epsilon_end=0.01,
                 C=1000000,
                 layer1_size=128, 
                 layer2_size=64, 
                 reproduce_seed=None
                ):
        
        #<------------> Env Specifications <------------->        
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.reproduce_seed=reproduce_seed
        # disable eager execution cause its slows down the training process         
        #tf.compat.v1.disable_eager_execution() 
        if self.reproduce_seed is not None:
            ## GLOBAL SEED ##  
            np.random.seed(self.reproduce_seed)
            tf.random.set_seed(self.reproduce_seed)
            self.env.seed(self.reproduce_seed)
            ## GLOBAL SEED ##  
        
        
        #Initialize environment related information        
        self.num_actions = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = [i for i in range(self.num_actions)]

        self.gamma = gamma        
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        
        self.B = B

        
        ##  Initialize a Replay Memory (D) in DQN agent## 
        self.D = ReplayMemory(C, self.state_space)

        
        '''
        Create and Initialize our Q_network
        Our Q-Network is intialized with two hidden layers
        Both the hidden layers have ReLu activations function
        '''

        self.Q = keras.Sequential([
        keras.layers.Dense(layer1_size, activation=tf.nn.relu),
        keras.layers.Dense(layer2_size, activation=tf.nn.relu),
        keras.layers.Dense(self.num_actions, activation=None)])
        self.Q.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha), loss=tf.keras.losses.MeanSquaredError())
        
        ##  book-keeping to store agent's training related information## 
        self.book_keeping = {
        'episodes': None,
        'rewards_per_ep': [],
        'mean_rewards_per_ep': [],
        'steps_per_ep': [],
        'eps_history': []
             
        }

        self.booking_keeping_df=None
        self.training_end_ep_index=None
        self.timestr=None
        self.trained=False
        
# This methods saves the experience tuple in Replay Memory D
    def save_experience(self, S, A, R, S_new, done):
        self.D.save_transition(S, A, R, S_new, done)

# Method choosen an action based on epsilon greedy strategy
    def epsilon_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state__ = np.array([state])
            actions = self.Q.predict(state__)
            action = np.argmax(actions)
        return action
    
    def update_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
# Method trains the Q-Network by sample a batch of experiences from D
    def learn(self):
        if self.D.C_counter < self.B:
            return

        S_js, A_js,  R_js,  S_nexts,  done_js = self.D.sample_mini_batch(self.B)

        Q_current = self.Q.predict(S_js) 
        Q_next = self.Q.predict( S_nexts)
        Q_Target = np.copy(Q_current)
        # update the Target Q_values as per our Q_target equation
        batch_index = np.arange(self.B, dtype=np.int32)
        Q_Target[batch_index, A_js] =  R_js + self.gamma * np.max(Q_next, axis=1)* done_js

        loss=self.Q.train_on_batch(S_js, Q_Target)
        
        self.update_epsilon()
        return loss

    # This method helps in training the network for a specific number f episodes    
    def train_multiple_episodes(self,num_episodes=500):

        for ep in range(num_episodes):  # this can be changed to train_multiple_episodes as a function            
            done = False
            ep_reward = 0
            S = self.env.reset()
            ep_steps=0
            
            while not done:

                A = self.epsilon_greedy_action(S)
                S_new , R, done , _ = self.env.step(A)
                ep_reward  += R
                ep_steps+=1
                self.save_experience(S, A, R, S_new , int(done))                
                S = S_new
                # Point - In DQN_Agent the learning takes place after every step
                self.learn()
               
            
            self.book_keeping['episodes']=ep+1
            self.book_keeping['rewards_per_ep'].append(ep_reward)
            mean_reward = np.mean(self.book_keeping['rewards_per_ep'][-100:])
            self.book_keeping['mean_rewards_per_ep'].append(mean_reward)
            self.book_keeping['steps_per_ep'].append(ep_steps)
            self.book_keeping['eps_history'].append(self.epsilon)
            
            print("\rEp: {} ,  Ep_Steps: {} ,Epsilon: {:.2f}, Ep_Reward : {:.2f} , Average_Reward : {:.2f}".format(
                                                                                self.book_keeping['episodes'],
                                                                                self.book_keeping['steps_per_ep'][-1],                                     
                                                                                self.book_keeping['eps_history'][-1],
                                                                                self.book_keeping['rewards_per_ep'][-1],
                                                                                self.book_keeping['mean_rewards_per_ep'][-1]), end="")
            
            if self.book_keeping['mean_rewards_per_ep'][-1] >= 500 and self.env_name=="CartPole-v1":
                print("\nMean Reward over last 100 ep more than 500")
                break
            if self.book_keeping['mean_rewards_per_ep'][-1] >= 200 and self.book_keeping['episodes']>=300 and  self.env_name=='LunarLander-v2':
                print("\nMean Reward over last 100 ep more than 200")
                break
        print("\n Agent trained.....")    
        self.trained=True


        print("\n Saving Model info.....")    
        self.save_training_info()    
        print("\n {} Problem took {} episodes".format(self.env_name,self.book_keeping['episodes']))
        # Get end episode number #


    def save_training_info(self):
        
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        dir_path = os.path.join("DQN_trained_models", self.env_name ,"model_"+self.timestr)
        os.makedirs(dir_path, exist_ok=True)
        model_path = os.path.join(dir_path, "model.h5")

        self.booking_keeping_df = pd.DataFrame({
                    'Rewards': self.book_keeping['rewards_per_ep'],
                    'Avg_Rewards': self.book_keeping['mean_rewards_per_ep'],
                    'Epsilon': self.book_keeping['eps_history'],
                    'Steps': self.book_keeping['steps_per_ep']
                    }, 
                    index= np.arange(self.book_keeping['episodes']))
        file_path = os.path.join(dir_path, "book_keeping.csv")
        self.booking_keeping_df.to_csv(file_path, sep='\t')
                
        
        self.save_model(path=model_path)

    def save_model(self,path=None):
        self.Q.save(path)


    def get_trained_model_info(self):
        #if self.Q not None and self.booking_keeping_df not None:
        return self.Q ,self.booking_keeping_df

    def load_pre_trained_model_info(self,timestr=None):
        dir_path = os.path.join("DQN_trained_models", self.env_name ,"model_"+timestr)
        model_path = os.path.join(dir_path, "model.h5")

        file_path = os.path.join(dir_path, "book_keeping.csv")

        # also get the images 
        images_path=os.path.join(dir_path, "IMAGES")
        images_paths=[]
        from os import listdir
        from os.path import isfile, join
        
        onlyimgfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))]
        images_paths = [os.path.join(dir_path, "IMAGES" , f) for f in onlyimgfiles]
        

        self.Q = keras.models.load_model(model_path)
        return self.Q , pd.read_csv(file_path, sep='\t') , images_paths
        #####

    def run_test_instances(self,case_list=None, model_=None):
        
        test_cases_data,image_paths= Rendering(env_name=self.env_name,
                 model=model_,
                 case_list=case_list,
                 timestr=self.timestr 
                 ).test_instances_of_env()
        return test_cases_data,image_paths