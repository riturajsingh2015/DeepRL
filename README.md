
## Docker 

![Docker_Logo](https://raw.githubusercontent.com/docker-library/docs/c350af05d3fac7b5c3f6327ac82fe4d990d8729c/docker/logo.png)

We created a docker image on the docker hub which contains all the completed code the related implementations, one may follow the commands below to pull and spin the container.

* **Note** : The trained models are present in the Docker image once you run the container.
---------------
Pull the image  
```
docker pull riturajsingh2015/deep_rl:v2.0
```
Spin a container


```
docker run −d −p 8888:8888  riturajsingh2015/deep_rl:v2.0
```
In case you need a token as a password to log in, copy the token from the container’s logs

When the container starts, please open your browser and type localhost:8888 to see Jupyter Interface with notebooks in Experiments folders and agents-implementation in agents folder.
---------------

# Applications of Deep Reinforcement Learning on Control System Environments
In this repository we have used two deep reinforcement learning algorithms on two 
control sytem simulation enviroments offered by openAI gym inorder to compare the performance across amoung the agents and setup some guidelines while dealing with these creating and using these reinforcement learning agents.

## Algorithms
The two algorithms that we analysed for this research include :
* Deep Q networks 
* Reinforce Policy Gradients

## Control System Enviroments
And the two control System Enviroments that we used in order to conduct our experiments include 
* Cartpole-v1
* LunarLander-v2

These two control systems simulation enviroments are offered by _OpenAI gym_ which is a toolkit for developing and comparing reinforcement learning algorithms. This toolkit supports teaching agents everything from walking to playing games like Pong or Pinball.

## Background

Deep Reinforcement Learning is a subsection in the field of Machine Learning and Artificial Intelligence.
In which there is an Agent and an environment. The Agent takes Action on the environment and receives a reward. The Goal of the Agent is to learn from the environment in turn maximizing the rewards.

### Elements of Reinforcement Learning

* Policy – describes Agent’s behavior at a given time
* Reward – Goodness at a given time
* Value function– Estimates Goodness in the long run
* Goal of Agent – Maximize cumulative Rewards 

## Motivation
The motivation for this research include : 
* Lack prior guidelines for choice of algorithm
* How to apply these algorithms or policies on different environment?
* Analyze which algorithm works better in a particular environment?
* Judge Agents behavior
* Mathematical insights of Algorithms

## Problem Statement

The problem statement for the research include
setting up guidelines for the choice of algorithms
two Reinforcement algorithms Deep Q Network which is a Tabular Method and uses Neural Network as Function Approximator
and the second algorithms used in experiment was Policy Gradient which is an Approximation Method
that Iterates over Policies to find the best
this method also uses Neural Network as Function Approximator. The second major problem which we want to answer to Judge Agent’s behavior and learning performance over different experimental setup.

## Experimental Setup
### Cartpole-v1

In the Cartpole-v1 environment there is a Pole which is attached by an un-actuated joint to a cart
that moves along a frictionless track
The goal of problem is to prevent Pole from Falling over 

![Cartpole-v1](https://www.oreilly.com/library/view/hands-on-q-learning-with/9781789345803/assets/9170409d-15f1-453b-816a-6f601a89fcf2.png)

*Fig. 2: This Photo was taken from oreilly.com*


**Environment specifications**
* Actions space – 2 
    * Left or Right
* Observations space – 4
    * Cart Position
    * Cart Velocity
    * Pole Angle
    * Pole Angular Velocity
* Reward - 1 points per step
* Episode Termination
    * Pole Angle > 15 deg
    * Episode last more than 500 steps

### LunarLander-v2
In the LunarLander-v2 environment there is a vehicle that starts from the top of the screen.
And there is a Landing pad that is always at coordinates (0,0).The goal of the problem is to  Land the vehicle on the target.

![LunarLander-v2](https://miro.medium.com/max/1346/1*i7lxpgt2K3Q8lgEPJu3_xA.png)

*Fig. 3: This Photo was taken from miro.medium.com*

**Environment specifications**
* Actions space – 4 
    * Left or Right or Main Engine or Do Nothing
* Observations space – 8
    * x coordinate of the lander
    * y coordinate of the lander
    * vx, the horizontal velocity
    * vy, the vertical velocity
    * θ, the orientation in space
    * vθ, the angular velocity
    * Left leg touching the ground (Boolean)
    * Right leg touching the ground (Boolean)

* Rewards = 100....140 
    * Ranging from the top of the  screen to landing pad
* Episode Termination
    * If Lander crashes or comes to rest on the landing pad. Receiving an additional -100 or +100 points.
    * Episode last more than 1000 steps

### Evalution Criteria for the Experiment
To Evaluate our experiment we will have the following performance metrics for our Agent consisting of - 
* Episodes required for sufficient training
* Average Steps to solve the problem
* Agent‘s Goodness to achieve the target which will Capture agent‘s Behaviour 
    * On multiple test instances of the environment



