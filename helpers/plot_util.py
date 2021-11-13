import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def beautify_axis(ax):
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # Customize the minor grid
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


def PG_learning_plot(arg):
    # Data for plotting

    x=np.arange(arg.shape[0])
    y1=arg['Rewards']
    y2=arg['Avg_Rewards']
    y3=arg['Steps']
    y4=arg['Loss']     
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 ,figsize=(16, 10))

    for ax in ax1, ax2 ,ax3 ,ax4:
        beautify_axis(ax)
    
    # plot lines
    ### ax1 ####
    ax1.plot(x, y1, label = "Rewards per ep" , color="blue")
    ax1.plot(x, y2, label = "Avg Rewards 100 ep" , color="red")
    ax1.legend()
    ax1.set(xlabel='Episodes', ylabel='Rewards')
    
    ### ax2 ####
    ax2.plot(x, y3, label = "Steps per ep" , color="darkred")
    ax2.legend()
    ax2.set(xlabel='Episodes', ylabel='Steps')

    ### ax3 ####
    ax3.plot(x, y3, label = "Steps per ep" , color="royalblue")
    ax3.legend()
    ax3.set(xlabel='Episodes', ylabel='Steps')
    
    ### ax4 ####
       
    ax4.plot(x, y4, label = "Loss" , color="purple")
    ax4.legend()
    ax4.set(xlabel='Episodes', ylabel='Loss')
    
    #fig.savefig('Eps_N_Reward.png')    
    
    plt.show()


def DQN_learning_plot(arg):
    # Data for plotting

    x=np.arange(arg.shape[0])
    y1=arg['Rewards']
    y2=arg['Avg_Rewards']
    y3=arg['Steps']
    y4=arg['Epsilon']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 ,figsize=(16, 10))

    for ax in ax1, ax2 ,ax3 ,ax4:
        beautify_axis(ax)
    
    # plot lines
    ### ax1 ####
    ax1.plot(x, y1, label = "Rewards per ep" , color="blue")
    ax1.plot(x, y2, label = "Avg Rewards 100 ep" , color="red")
    ax1.legend()
    ax1.set(xlabel='Episodes', ylabel='Rewards')
    
    ### ax2 ####
    ax2.plot(x, y3, label = "Steps per ep" , color="darkred")
    ax2.legend()
    ax2.set(xlabel='Episodes', ylabel='Steps')

    ### ax3 ####
    ax3.plot(x, y3, label = "Steps per ep" , color="royalblue")
    ax3.legend()
    ax3.set(xlabel='Episodes', ylabel='Steps')
    
    ### ax4 ####
       
    ax4.plot(x, y4, label = "Epsilon" , color="purple")
    ax4.legend()
    ax4.set(xlabel='Episodes', ylabel='Epsilon')
    
    #fig.savefig('Eps_N_Reward.png')    
    
    plt.show()
    
    
def plot_test_cases(df):
    df.plot.bar(rot=0,figsize=(12, 5))


    