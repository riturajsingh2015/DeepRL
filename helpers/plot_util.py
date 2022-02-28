import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def beautify_axis(ax):
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.4', color='lightgrey')
        # Customize the minor grid
        ax.grid(which='minor', linestyle=':', linewidth='0.4', color='silver')


def PG_learning_plot(arg):
    # Data for plotting

    x=np.arange(arg.shape[0])
    y1=arg['Rewards']
    y2=arg['Avg_Rewards']
    y3=arg['Steps']
    y4=arg['Loss']     
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 ,figsize=(16, 8), sharex=True)

    for ax in (ax1,ax2,ax3,ax4):
        beautify_axis(ax)
    
    # left_subplot
    ax1.plot(x, y1, label = "Rewards per Episode" , color="r")
    ax1.legend(loc=4)
    ax2.plot(x, y2, label = "Avg Rewards 100 Episodes" , color="b")
    ax2.legend(loc=4)
    #l_plot.set(xlabel='Episodes', ylabel='Rewards')
    ax3.plot(x, y3, label = "Steps per Episode" , color="m")
    ax3.legend(loc=4)
    
    ax4.plot(x, y4, label = "Training Loss per Episode" , color="y")
    ax4.legend(loc=4)
    plt.subplots_adjust(hspace=0.1)
    plt.show()


def DQN_learning_plot(arg):
    # Data for plotting

    x=np.arange(arg.shape[0])
    y1=arg['Rewards']
    y2=arg['Avg_Rewards']
    y3=arg['Steps']
    y4=arg['Epsilon']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 ,figsize=(14, 8), sharex=True)

    for ax in (ax1,ax2,ax3,ax4):
        beautify_axis(ax)
    
    # left_subplot
    ax1.plot(x, y1, label = "Rewards per Episode" , color="r")
    ax1.legend(loc=4)
    ax2.plot(x, y2, label = "Avg Rewards 100 Episodes" , color="b")
    ax2.legend(loc=4)
    #l_plot.set(xlabel='Episodes', ylabel='Rewards')
    ax3.plot(x, y3, label = "Steps per Episode" , color="m")
    ax3.legend(loc=4)
    
    ax4.plot(x, y4, label = "Epsilon per Episode" , color="y")
    ax4.legend(loc=4)
    plt.subplots_adjust(hspace=0.1)
    plt.show()
    
    
def plot_test_cases(df):
    ax = df.plot.bar(rot=0,figsize=(12, 4))
    ax.set_xlabel("Test-instance-Values")
    ax.legend(loc=3)  
    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID    
    beautify_axis(ax)
    