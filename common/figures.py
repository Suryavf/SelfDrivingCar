from os      import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import datetime
import os


""" Save plot
    ---------
    Args:
        data: Data to plot
        path: File to save

    Return: files list
"""
def saveScatterSteerSpeed(steer,speed,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            c = cmd[idx]
            axs[i,j].scatter(steer[command==c],speed[command==c],alpha=0.1)
            axs[i,j].grid(True)
            axs[i,j].set_xlabel("Steer")
            axs[i,j].set_ylabel("Speed")
            axs[i,j].set_title(hgl[idx])
            axs[i,j].set_xlim(-1.2,1.2)
            axs[i,j].set_ylim( -20, 90)
            idx += 1
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    fig.savefig(path)
    plt.close('all')
def saveScatterError(steer,steerErr,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            c = cmd[idx]
            axs[i,j].scatter(steer[command==c],steerErr[command==c],alpha=0.1)
            axs[i,j].grid(True)
            axs[i,j].set_xlabel("Steer (True)")
            axs[i,j].set_ylabel("Steer Error")
            axs[i,j].set_title(hgl[idx])
            axs[i,j].set_xlim(-1.2,1.2)
            axs[i,j].set_ylim(-0.1,1.2)
            idx += 1
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    fig.savefig(path)
    plt.close('all')
def saveScatterPolarSteerSpeed(steer,speed,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)
    steer = steer.reshape(-1)
    speed = speed.reshape(-1)

    x = speed*np.cos(steer)
    y = speed*np.sin(steer)
    for i in range(2):
        for j in range(2):
            c = cmd[idx]
            axs[i,j].scatter(y[command==c],x[command==c],alpha=0.05)
            axs[i,j].grid(True)
            axs[i,j].set_xlabel("y")
            axs[i,j].set_ylabel("x")
            axs[i,j].set_title(hgl[idx])
            axs[i,j].set_ylim(-20,90)
            axs[i,j].set_xlim(-50,50)
            idx += 1
    fig.tight_layout()
    fig.set_size_inches(10,10)
    fig.savefig(path)
    plt.close('all')

""" Save histogram
    --------------
    Args:
        data: Data to histogram
        path: File to save

    Return: files list
"""
def saveHistogramSteerSpeed(steer,speed,path):
    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)

    axs[0].hist(x=steer, bins=180)
    axs[0].set_xlim(-1.3,1.3)
    axs[0].set_ylim(0,18000)
    axs[0].set_title("Steer")

    axs[1].hist(x=speed, bins=180)
    axs[1].set_xlim(-5,90)
    axs[1].set_ylim(0,15000)
    axs[1].set_title("Speed")

    fig.set_size_inches(12,10)
    fig.savefig(path)
    plt.close('all')

def saveHistogramSteer(steer,path):
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    axs.hist(x=steer, bins=180)
    axs.set_xlim(-1.3,1.3)
    axs.set_ylim(0,18000)
    axs.set_title("Steer")

    fig.set_size_inches(12,5)
    fig.savefig(path)
    plt.close('all')


class savePlotByStep():
    def __init__(self, path,name):
        self._path = path
        self._name = name
        self._values = list()

    def update(self,values):
        # Add values 
        self._values.append(values)
        n_epoch = len(self._values)

        if n_epoch > 2:
            epochs = list( range(1,n_epoch+1) )
            name = self._name + ".svg"
            path = join(self._path,name)
            
            fig, ax = plt.subplots()
            ax.plot(epochs,self._values)
            plt.xlabel('Epoch')
            plt.ylabel(self._name)
            plt.xlim(1,n_epoch)
            plt.savefig(path)
            plt.close('all')

    def reset(self):
        self._values = list()

class save2PlotByStep():
    def __init__(self, path,name,line1,line2):
        self._path = path
        self._name = name
        self._line1 = line1
        self._line2 = line2
        self._values1 = list()
        self._values2 = list()

    def update(self,val1,val2):
        # Add values 
        self._values1.append(val1)
        self._values2.append(val2)
        n_epoch = len(self._values1)

        if n_epoch > 2:
            epochs = list( range(1,n_epoch+1) )
            name = self._name + ".svg"
            path = join(self._path,name)
            
            fig, ax = plt.subplots()
            ax.plot(epochs,self._values1)
            ax.plot(epochs,self._values2)

            plt.legend([self._line1,self._line2])
            plt.xlabel("Epoch")
            plt.savefig(path)
            plt.close('all')

    def reset(self):
        self._values1 = list()
        self._values2 = list()
        
