import tensorflow.keras as keras
from tensorflow.keras.layers import Dense , Activation, Input

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import gym

class PG_Agent(object):
    def __init__(self,
                 env_name="CartPole-v1",
                 ALPHA=0.0005,
                 GAMMA=0.99,
                 layer1_size=16, 
                 layer2_size=16,
                 fname='reinforce.h5',
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
        
        self.input_dims = self.env.observation_space.shape[0]#input_dims
        self.n_actions = self.env.action_space.n#n_actions
        #print(self.input_dims ,"      ", self.n_actions)
        
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(self.n_actions)]
        
        
        self.model_file = fname
        
        self.rewards_per_ep = []
        self.mean_rewards_per_ep=[]
        self.steps_per_ep=[]
        self.losses=[]
        self.training_end_ep_index=None
    
    
    

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

        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

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
            
            
            cost=self.learn()
            
            self.rewards_per_ep.append(ep_reward)                        
            mean_reward = np.mean(self.rewards_per_ep[-100:])
            self.mean_rewards_per_ep.append(mean_reward)
            self.steps_per_ep.append(ep_steps)
            self.losses.append(cost)
            print("\rEps: {} ,  Eps steps: {} , Ep_Reward : {:.2f} , Avg_Reward : {:.2f} , Loss: {:.2f}".format(ep,
                                                                                self.steps_per_ep[-1],                                     
                                                                                self.rewards_per_ep[-1],
                                                                                self.mean_rewards_per_ep[-1],
                                                                                self.losses[-1]), end="")
            
            self.training_end_ep_index=ep
            if self.mean_rewards_per_ep[-1] >= 500 and self.env_name=="CartPole-v1":
                print("\nMean Reward over last 100 ep more than 500")
                break
            if self.mean_rewards_per_ep[-1] >= 200 and self.env_name=='LunarLander-v2':
                print("\nMean Reward over last 100 ep more than 200")
                break
            
        print("\n {} Problem took {} episodes".format(self.env_name,self.training_end_ep_index+1))

    
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