import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import pandas as pd
import gym
from helpers.render_model import *

import datetime



'''
*****Notations used *********
agent's training related hyperparameters 
Learning rate     - alpha (α), 
Discount factor   - gamma (γ), 
policy-network    - pi (π) related 
Number of Hidden layers in the policy-network = 2
Number of neurons in layer 1 - layer1_size
Number of neurons in layer 2 - layer2_size

trajectory        - tau (τ),


# agent's transition related 
S     -   current state
S_new -   new state
A     -   action
R     -   reward


'''



class PG_Agent(object):
    def __init__(self,
                 env_name="CartPole-v1",
                 ALPHA=0.0005,
                 GAMMA=0.98,
                 layer1_size=16, 
                 layer2_size=16,
                 sol_th=495,
                 reproduce_seed=None
                ):
        
        #<------------> Env Specifications <------------->        
        self.env_name = env_name
        #self.reward_div = 100 if self.env_name=='LunarLander-v2' else 1
        self.env = gym.make(self.env_name)
        self.reproduce_seed=reproduce_seed        
        if self.reproduce_seed is not None:
            ## GLOBAL SEED ##  
            #print("setting global seed {}".format(self.reproduce_seed))
            np.random.seed(self.reproduce_seed)
            tf.random.set_seed(self.reproduce_seed)
            self.env.seed(self.reproduce_seed)
            ## GLOBAL SEED ##  
        
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        
        self.sol_th = sol_th
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        
        
        self.optimizer = tf.optimizers.Adam(self.alpha)       
  

        self.pi = keras.Sequential([
        keras.Input(shape=(self.state_space,), name="inputs"),
        keras.layers.Dense(layer1_size, activation='relu', kernel_initializer=keras.initializers.he_normal(),autocast=False),
        keras.layers.Dense(layer2_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(self.action_space, activation='softmax')])
        
        

        # Trajectory related
        self.tau = {
        'states': [],
        'actions': [],
        'rewards': []
        }
                
        
        ##  book-keeping dict ## 
        self.book_keeping = {
        'episodes': None,
        'rewards_per_ep': [],
        'mean_rewards_per_ep': [],
        'steps_per_ep': [],
        'loss': []   
        }

        self.booking_keeping_df=None
        
        self.dir_path= None
        self.model_path = None
        self.book_keeping_file_path = None
        
        self.training_time  = None
        
        

    def pg_loss(self,aprobs, actions, G):
        indexes = tf.range(0, tf.shape(aprobs)[0]) * tf.shape(aprobs)[1] + actions
        responsible_outputs = tf.gather(tf.reshape(aprobs, [-1]), indexes)
        loss = -tf.reduce_mean(tf.math.log(responsible_outputs) * G)
        return loss
    
    def choose_action(self, state):
        softmax_out = self.pi(state.reshape((1, -1)))
        selected_action = np.random.choice(self.action_space, p=softmax_out.numpy()[0])
        return selected_action

    def update_policy_network(self,states,actions,G):        
        with tf.GradientTape() as tape:
            aprobs = self.pi(states)
            loss = self.pg_loss(aprobs, actions, G)
        gradients = tape.gradient(loss, self.pi.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.pi.trainable_variables))
        return loss
            

    def get_discounted_rewards(self,rewards):
        reward_sum = 0
        G = []
        for reward in rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.gamma * reward_sum
            G.append(reward_sum)
        G.reverse()
        G = np.array(G)

        # standardise the rewards
        G -= np.mean(G)
        G /= np.std(G)
        return G


    def learn(self): 
        states, actions, rewards = self.tau['states'],self.tau['actions'],self.tau['rewards']
        G=self.get_discounted_rewards(rewards)        
        return self.update_policy_network(np.vstack(states) ,actions,G).numpy()
    
    def empty_the_trajectory(self):
        self.tau['states'] = []
        self.tau['actions'] = []
        self.tau['rewards'] = []
        
    def add_experience_to_trajectory(self,S,R,A):
        self.tau['states'].append(S)                   
        self.tau['rewards'].append(R)                
        self.tau['actions'].append(A)   
        
    def store_book_keeping(self,ep,ep_reward,ep_steps,loss):
        self.book_keeping['episodes']=ep+1
        self.book_keeping['rewards_per_ep'].append(ep_reward)
        self.book_keeping['mean_rewards_per_ep'].append(np.mean(self.book_keeping['rewards_per_ep'][-100:]))
        self.book_keeping['steps_per_ep'].append(ep_steps)
        self.book_keeping['loss'].append(loss)
    
    def print_book_keeping_info(self):
        print("\rEp: {} , Ep_Steps: {} , Ep_Reward : {:.2f} , Avg_Reward : {:.2f} , Loss: {:.2f}".
              format(self.book_keeping['episodes'],
                     self.book_keeping['steps_per_ep'][-1],
                     self.book_keeping['rewards_per_ep'][-1],
                     self.book_keeping['mean_rewards_per_ep'][-1],
                     self.book_keeping['loss'][-1]), end="")
        
    def is_solved(self):
        if self.book_keeping['mean_rewards_per_ep'][-1] >= self.sol_th:
            print("\nMean Reward over last 100 ep more than {}".format(self.sol_th))
            return True
        return False    

    
    def train_multiple_episodes(self,num_episodes=500):
        training_start_time = datetime.datetime.now()
        
        for ep in range(num_episodes):  # this can be changed to train_multiple_episodes as a function
            ep_reward = 0
            ep_steps=0
            S = self.env.reset()
            self.empty_the_trajectory()
            
            done =False
            while not done:
                A = self.choose_action(S)
                S_new, R, done, _ = self.env.step(A)
                #store experience in the trajectory - tau
                self.add_experience_to_trajectory(S,R,A)                
                #make transition to a new state
                S = S_new              
                ep_reward += R
                ep_steps+=1
                
            # make the policy network learn at the end of episode 
            loss=self.learn()
                
            self.store_book_keeping(ep,ep_reward,ep_steps,loss)            
            self.print_book_keeping_info()
            
            if self.is_solved():
                training_end_time = datetime.datetime.now()
                self.training_time= training_end_time-training_start_time
                self.save_training_info(ep)                
                break
                
                
    def set_paths(self,dir_path):
        self.dir_path=dir_path
        self.model_path = os.path.join(self.dir_path, "model.h5")
        self.book_keeping_file_path = os.path.join(self.dir_path, "book_keeping.csv")
                
                

    def save_training_info(self,ep):        
        time_stamp = str(self.training_time) #time.strftime("%Y%m%d-%H%M%S")
        model_name="{}__{}__{}__({}*{})__{}____{}".format(ep,self.alpha,
                                                            self.gamma,self.layer1_size,self.layer2_size,self.sol_th,time_stamp) 
        dir_path = os.path.join("..","PG_trained_models", self.env_name ,model_name)
        os.makedirs(dir_path, exist_ok=True)
        

        self.booking_keeping_df = pd.DataFrame({
                    'Rewards': self.book_keeping['rewards_per_ep'],
                    'Avg_Rewards': self.book_keeping['mean_rewards_per_ep'],
                    'Loss': self.book_keeping['loss'],
                    'Steps': self.book_keeping['steps_per_ep']
                    }, 
                    index= np.arange(self.book_keeping['episodes']))
        
        self.set_paths(dir_path)        
        self.save_model_info()
       


    def save_model_info(self):        
        self.booking_keeping_df.to_csv(self.book_keeping_file_path, sep='\t')             
        self.pi.save_weights(self.model_path)
        print("\n Model Saved at {}".format(self.dir_path))    
        
        
    def load_model_by_dir(self , dir_path=None):
        self.set_paths(dir_path) 
        load_status=self.pi.load_weights(self.model_path)
        return self.pi, pd.read_csv(self.book_keeping_file_path, sep='\t')    
   


    def run_test_instances(self,case_list=None, model_ref=None):        
        test_cases_data,image_paths= Rendering(env_name=self.env_name,
                 test_model=model_ref,
                 case_list=case_list,
                 dir_path=self.dir_path
                 ).test_instances_of_env()
        return test_cases_data,image_paths


