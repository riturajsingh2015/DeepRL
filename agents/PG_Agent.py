import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense , Activation, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import os
import time
import pandas as pd
import gym
from helpers.render_model import *
#from helpers.plot_util import *

class PG_Agent(object):
    def __init__(self,
                 env_name="CartPole-v1",
                 ALPHA=0.0005,
                 GAMMA=0.99,
                 layer1_size=16, 
                 layer2_size=16,
                 #fname='reinforce.h5',
                 reproduce_seed=None
                ):
        
        #<------------> Env Specifications <------------->        
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.reproduce_seed=reproduce_seed        
        tf.compat.v1.disable_eager_execution() 
        tf.python.util.deprecation._PRINT_DEPRECATION_WARNINGS = False
        if self.reproduce_seed is not None:
            ## GLOBAL SEED ##  
            #print("setting global seed {}".format(self.reproduce_seed))
            np.random.seed(self.reproduce_seed)
            tf.random.set_seed(self.reproduce_seed)
            self.env.seed(self.reproduce_seed)
            ## GLOBAL SEED ##  
        
        self.n_actions = self.env.action_space.n#n_actions
        self.input_dims = self.env.observation_space.shape[0]#input_dims
        self.action_space = [i for i in range(self.n_actions)]
        
        
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        
        # Model related stuff
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        

        self.policy, self.predict = self.build_policy_network()

        # Memory related
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

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
        
        #self.model_file = fname
    

    def build_policy_network(self):
        #tf.compat.v1.disable_eager_execution()
        input_ = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input_)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)

        # y_true will be target labels "actions" received from the train_on_batch function
        # y_pred will be the predicted probabilities from the model
        def custom_loss(y_true, y_pred): 
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)  

            return K.sum(-log_lik*advantages)

        policy = Model(inputs=[input_, advantages], outputs=[probs])

        policy.compile(optimizer=Adam(learning_rate=self.lr), loss=custom_loss)

        predict = Model(inputs=[input_], outputs=[probs])

        return policy, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        
        # one hot encode the actions
        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1
        
        
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std
            
        
        cost = self.policy.train_on_batch([state_memory, self.G], actions)  # x=training inputs and y= target outputs

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return cost
    
    def train_multiple_episodes(self,num_episodes=500):
        for ep in range(num_episodes):  # this can be changed to train_multiple_episodes as a function
            done = False
            ep_reward = 0
            ep_steps=0
            observation = self.env.reset()
            while not done:
                action = self.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                self.store_transition(observation, action, reward)
                observation = observation_                
                ep_reward += reward
                ep_steps+=1
            
            
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
            
            if self.book_keeping['mean_rewards_per_ep'][-1] >= 500 and self.env_name=="CartPole-v1":
                print("\nMean Reward over last 100 ep more than 500")
                break
            if self.book_keeping['mean_rewards_per_ep'][-1] >= 200 and self.env_name=='LunarLander-v2':
                print("\nMean Reward over last 100 ep more than 200")
                break
        print("\n Agent trained.....")    
        self.trained=True


        print("\n Saving Model info.....")    
        self.save_training_info()    
        print("\n {} Problem took {} episodes".format(self.env_name,self.book_keeping['episodes']))
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
        self.predict.save_weights(path)

    def get_trained_model_info(self):
        return self.predict , self.booking_keeping_df

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







##______________________________________######
'''
    def get_stats(self):
        num_episodes=self.training_end_ep_index+1
        return num_episodes,self.rewards_per_ep,self.mean_rewards_per_ep,self.steps_per_ep,self.losses   
    
    def get_trained_model(self):
        return self.predict
    
    def save_model_weights(self):
        self.predict.save_weights('./checkpoints/my_checkpoint')

    def load_default_model(self):
        # Create a new model instance
        _, model_predict = self.build_policy_network()
        # Restore the weights
        model_predict.load_weights('./checkpoints/my_checkpoint')
        
        if self.env_name=="CartPole-v1":
            x = model_predict
        if self.env_name=='LunarLander-v2':
            x = model_predict
        return x

'''