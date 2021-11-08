import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import os
import time
import pandas as pd

class ReplayMemory():
    # this memory is required to save the transitions an later sample
    # transitions to reduce the network losses
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def save_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal



class DQN_Agent():
    def __init__(self, 
                 lr=0.001, 
                 env_name="CartPole-v1",                 
                 gamma=0.98, 
                 epsilon=1.0, 
                 batch_size=64,
                 epsilon_dec=1e-3, 
                 epsilon_end=0.01,
                 mem_size=1000000,
                 layer1_size=128, 
                 layer2_size=64, 
                 fname='dqn_model.h5',
                 reproduce_seed=None
                ):
        
        #<------------> Env Specifications <------------->        
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.reproduce_seed=reproduce_seed        
        tf.compat.v1.disable_eager_execution() 
        if self.reproduce_seed is not None:
            ## GLOBAL SEED ##  
            #print("setting global seed {}".format(self.reproduce_seed))
            np.random.seed(self.reproduce_seed)
            tf.random.set_seed(self.reproduce_seed)
            self.env.seed(self.reproduce_seed)
            ## GLOBAL SEED ##  
        
        
        
        self.n_actions = self.env.action_space.n
        self.input_dims = self.env.observation_space.shape
        
        #<------------> Reproduce Seed       <-----------> 
    
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        
        self.batch_size = batch_size
        self.model_file = fname
        
        ##  Replay Memory ## 
        self.memory = ReplayMemory(mem_size, self.input_dims)
        #self.q_eval = create_dqn_nn_4(lr, self.n_actions, self.input_dims)
        
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        
        self.q_eval = keras.Sequential([
        keras.layers.Dense(self.fc1_dims, activation='relu'),
        keras.layers.Dense(self.fc2_dims, activation='relu'),
        keras.layers.Dense(self.n_actions, activation=None)])
        self.q_eval.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
        
        ##  book-keeping ## 
        self.book_keeping = {
        'episodes': None,
        'rewards_per_ep': [],
        'mean_rewards_per_ep': [],
        'steps_per_ep': [],
        'eps_history': []
             
        }

        self.booking_keeping_df=None
        self.training_end_ep_index=None
        
        

    def save_transition(self, state, action, reward, new_state, done):
        self.memory.save_transition(state, action, reward, new_state, done)

    def epsilon_greedy_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            #print("Random action taken---------")
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
            #print("NN action taken---------")

        return action
    
    def update_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # sample a batch from the replay memory for network updation
        # states_ is the observation received from the enviroment once corresponding action
        # was taken on the state
        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)
        '''
        print("states.........")
        print(states, states.shape , end="\n\n")
        print("actions.........")
        print(actions, actions.shape , end="\n\n")
        print("rewards.........")
        print(rewards, rewards.shape , end="\n\n")
        '''
        q_eval = self.q_eval.predict(states)
        #print("q_eval.........")
        #print(q_eval, q_eval.shape , end="\n\n")
        
        q_next = self.q_eval.predict(states_)

        # get initial q_target using predicted values from the network
        q_target = np.copy(q_eval)

        #print("q_target.........")
        #print(q_target, q_target.shape , end="\n\n")
        
        #input()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # update q_target = reward + gamma * maximum amoung all next predicted states
        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones
        
        #print("q_target.........")
        #print(q_target, q_target.shape , end="\n\n")
        #input()
        cost=self.q_eval.train_on_batch(states, q_target)
        '''
        self.q_eval.fit(states, 
                        q_target,
                        epochs=5,
                        batch_size=self.batch_size,
                        verbose=0)
        '''
        
        self.update_epsilon()
        return cost

        
    def train_multiple_episodes(self,num_episodes=500):

        for ep in range(num_episodes):  # this can be changed to train_multiple_episodes as a function            
            done = False
            ep_reward = 0
            observation = self.env.reset()
            ep_steps=0
            
            while not done:
                action = self.epsilon_greedy_action(observation)
                observation_, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_steps+=1
                self.save_transition(observation, action, reward, observation_, int(done))
                observation = observation_
                # make it learn at everytimetamp
                self.learn()
               
            
            #self.learn()
            self.book_keeping['episodes']=ep+1
            self.book_keeping['rewards_per_ep'].append(ep_reward)
            mean_reward = np.mean(self.book_keeping['rewards_per_ep'][-100:])
            self.book_keeping['mean_rewards_per_ep'].append(mean_reward)
            self.book_keeping['steps_per_ep'].append(ep_steps)
            self.book_keeping['eps_history'].append(self.epsilon)
            
            print("\rEps: {} ,  Eps steps: {} ,Epsilon: {:.2f}, Ep_Reward : {:.2f} , Average_Reward : {:.2f}".format(
                                                                                self.book_keeping['episodes'],
                                                                                self.book_keeping['steps_per_ep'][-1],                                     
                                                                                self.book_keeping['eps_history'][-1],
                                                                                self.book_keeping['rewards_per_ep'][-1],
                                                                                self.book_keeping['mean_rewards_per_ep'][-1]), end="")
            
            if self.book_keeping['mean_rewards_per_ep'][-1] >= 505 and self.env_name=="CartPole-v1":
                print("\nMean Reward over last 100 ep more than 600")
                break
            if self.book_keeping['mean_rewards_per_ep'][-1] >= 210 and self.env_name=='LunarLander-v2':
                print("\nMean Reward over last 100 ep more than 300")
                break

        print("Saving Model info.....")    
        self.save_training_info()    
        print("\n {} Problem took {} episodes".format(self.env_name,self.book_keeping['episodes']))
        # Get end episode number


    def save_training_info(self):
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        dir_path = os.path.join("DQN_trained_models", self.env_name ,"model_"+timestr)
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
        self.q_eval.save(path)


    def get_trained_model_info(self):
        #if self.q_eval not None and self.booking_keeping_df not None:
        return self.q_eval ,self.booking_keeping_df

    def load_pre_trained_model_info(self,timestr=None):
        dir_path = os.path.join("DQN_trained_models", self.env_name ,"model_"+timestr)
        model_path = os.path.join(dir_path, "model.h5")

        file_path = os.path.join(dir_path, "book_keeping.csv")

        self.q_eval = keras.models.load_model(model_path)
        return self.q_eval , pd.read_csv(file_path, sep='\t')