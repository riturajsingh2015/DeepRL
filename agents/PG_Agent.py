import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import pandas as pd
import gym
from helpers.render_model import *



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
        self.gamma = GAMMA
        self.optimizer = tf.optimizers.Adam(ALPHA)       
  

        self.pi = keras.Sequential([
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
        self.training_end_ep_index=None
        self.timestr=None
        self.trained=False
        

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
        loss=self.update_policy_network(np.vstack(states) ,actions,G)
        return loss
    
    def empty_the_trajectory(self):
        self.tau['states'] = []
        self.tau['actions'] = []
        self.tau['rewards'] = []
        
    def add_experience_to_trajectory(self,S,R,A):
        self.tau['states'].append(S)                   
        self.tau['rewards'].append(R)                
        self.tau['actions'].append(A)         

    
    def train_multiple_episodes(self,num_episodes=500):
        
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
                
            
            self.book_keeping['episodes']=ep+1
            self.book_keeping['rewards_per_ep'].append(ep_reward)
            mean_reward = np.mean(self.book_keeping['rewards_per_ep'][-100:])
            self.book_keeping['mean_rewards_per_ep'].append(mean_reward)
            self.book_keeping['steps_per_ep'].append(ep_steps)
            self.book_keeping['loss'].append(loss)
            
            print("\rEp: {} , Ep_Steps: {} , Ep_Reward : {:.2f} , Avg_Reward : {:.2f} , Loss: {:.2f}".format(
                                                                                self.book_keeping['episodes'],
                                                                                self.book_keeping['steps_per_ep'][-1],                                     
                                                                                self.book_keeping['rewards_per_ep'][-1],
                                                                                self.book_keeping['mean_rewards_per_ep'][-1],
                                                                                self.book_keeping['loss'][-1]), end="")
            
            if self.book_keeping['mean_rewards_per_ep'][-1] >= self.sol_th and self.env_name=="CartPole-v1":
                print("\nMean Reward over last 100 ep more than {}".format(self.sol_th))
                break
            if self.book_keeping['mean_rewards_per_ep'][-1] >= self.sol_th and self.env_name=='LunarLander-v2':
                print("\nMean Reward over last 100 ep more than {}".format(self.sol_th))
                break
        #print("\n Agent trained.....")    
        self.trained=True


        #print("\n Saving Model info.....")    
        self.save_training_info()    
        #print("\n {} Problem took {} episodes".format(self.env_name,self.book_keeping['episodes']))
        # Get end episode number


    def save_training_info(self):
        
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        dir_path = os.path.join("PG_trained_models", self.env_name ,"model_"+self.timestr)
        os.makedirs(dir_path, exist_ok=True)
        model_path = os.path.join(dir_path, "model.h5")

        self.booking_keeping_df = pd.DataFrame({
                    'Rewards': self.book_keeping['rewards_per_ep'],
                    'Avg_Rewards': self.book_keeping['mean_rewards_per_ep'],
                    'Loss': self.book_keeping['loss'],
                    'Steps': self.book_keeping['steps_per_ep']
                    }, 
                    index= np.arange(self.book_keeping['episodes']))
        file_path = os.path.join(dir_path, "book_keeping.csv")
        self.booking_keeping_df.to_csv(file_path, sep='\t')                
        
        self.save_model(path=model_path)

    #def plot_learning_curves(self):
    #    PG_learning_plot(self.book_keeping)

    def save_model(self,path=None):
        self.pi.save(path)
        print("Model saved")

    def get_trained_model_info(self):
        return self.pi , self.booking_keeping_df

    def load_pre_trained_model_info(self,timestr=None):
        dir_path = os.path.join("PG_trained_models", self.env_name ,"model_"+timestr)
        model_path = os.path.join(dir_path, "model.h5")

        file_path = os.path.join(dir_path, "book_keeping.csv")

        # also get the images
        images_path=os.path.join(dir_path, "IMAGES")
        images_paths=[]
        from os import listdir
        from os.path import isfile, join
        
        onlyimgfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))]
        images_paths = [os.path.join(dir_path, "IMAGES" , f) for f in onlyimgfiles]
        
        _,self.predict=self.build_policy_network()
        self.predict.load_weights(model_path) #= keras.models.load_model(model_path)
        return self.predict , pd.read_csv(file_path, sep='\t') , images_paths


    def run_test_instances(self,case_list=None, model_=None):        
        test_cases_data,image_paths= Rendering(env_name=self.env_name,
                 model=model_,
                 case_list=case_list,
                 timestr=self.timestr 
                 ).test_instances_of_env()
        ###print(test_cases_data,image_paths)
        return test_cases_data,image_paths


