import os
import glob
import argparse
import datetime
from random import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

""" Cooked files list
    -----------------
    Args:
        path: Files directory
        mode: Train or Valid

    Return: files list
"""
def cookedFilesList(path,mode):
    files = [os.path.join(path,f) for f in os.listdir(path) 
                                        if os.path.isfile(os.path.join(path,f)) 
                                                                and mode in f]
    if len(files)>1:    
        shuffle(files)        
    else:
        files = files[0]

    return files


""" Save plot
    ---------
    Args:
        data: Data to plot
        path: File to save

    Return: files list
"""
def savePlot(data,title,path):
    n_lines = len(title)

    if n_lines == 1:
        epochs = np.arange(1,len(data)+1)
        plt.plot(epochs,data)
        plt.xlabel("Epoch")
        plt.ylabel(title[0])

    else:
        epochs = np.arange(1,len(data[0])+1)
        fig, ax = plt.subplots()
        for i in range(n_lines):
            ax.plot(epochs,data[i])
        plt.legend(title)
        plt.xlabel("Epoch")
        plt.savefig(path)

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

def saveScatterPolarSteerSpeed(steer,speed,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)

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
    plt.show()
    fig.savefig(path)


""" Save histogram
    --------------
    Args:
        data: Data to histogram
        path: File to save

    Return: files list
"""
def saveHistogram(data,path):
    n, bins, patches = plt.hist(x=data, bins=60)#, color='#0504aa',
                               # alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    plt.ylim(ymax=maxfreq)#np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.xlim(-2,2)
    plt.savefig(path)
def saveHistogramSteerSpeed(steer,speed,path):
    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)

    axs[0].hist(x=steer, bins=180)
    axs[0].set_ylim(0,18000)
    axs[0].set_title("Steer")

    axs[1].hist(x=speed, bins=180)
    axs[1].set_xlim(-5,90)
    axs[1].set_ylim(0,15000)
    axs[1].set_title("Speed")

    fig.savefig(path)


""" Name directory model
    --------------------
    Args:
        mode: Train or Valid

    Return: Name for directory model
"""
def nameDirectoryModel(mode):
    # Time
    now = datetime.datetime.now()

    year  = str(now.year)[2:]
    if now.day<10: day = "0"+str(now.day)
    else         : day =     str(now.day)
    if now.hour<10: hour = "0"+str(now.hour)
    else          : hour =     str(now.hour)
    if now.month<10: month = "0"+str(now.month)
    else           : month =     str(now.month)
    if now.minute<10: minute = "0"+str(now.minute)
    else            : minute =     str(now.minute)
    
    time = year+month+day+hour+minute
    return mode+time
def nameDirectory():
    # Time
    now = datetime.datetime.now()

    year  = str(now.year)[2:]
    if now.day<10: day = "0"+str(now.day)
    else         : day =     str(now.day)
    if now.hour<10: hour = "0"+str(now.hour)
    else          : hour =     str(now.hour)
    if now.month<10: month = "0"+str(now.month)
    else           : month =     str(now.month)
    if now.minute<10: minute = "0"+str(now.minute)
    else            : minute =     str(now.minute)
    
    return year+month+day+hour+minute


""" String to bool
    ---------------
    Args:
        v: directory by check

"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if   v.lower() in ('yes', 'true','t','y','1'):
        return True
    elif v.lower() in ( 'no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


""" Check directory
    ---------------
    Args:
        d: directory by check

"""
def checkdirectory(d):
    if not os.path.exists(d):
        os.makedirs(d)
    

""" Averager class
    --------------
    Average of array or scalar
"""
class averager():
    def __init__(self,n=0):
        if n >0:
            self.mean = np.array(n)
        else:    
            self.mean  = 0
        self.count = 0
        self.n = n
    def reset(self):
        if self.n >0:
            self.mean = np.array(self.n)
        else:    
            self.mean  = 0
        self.count = 0
    def update(self,val):
        n = self.count
        self.count = n + 1
        self.mean  = (self.mean*n + val)/self.count
    def val(self):
        return self.mean


""" Counter class
    --------------
    Simple counter
"""      
class counter():
    def __init__(self):
        self.val = 0
    def reset(self):
        self.val = 0
    def update(self):
        self.val+= 1
       

""" Big Dictionary
    --------------
    Stack data to dictionary
"""  
class BigDict():
    def __init__(self):
        self.dictList = list()
    def update(self,dict_):
        self.dictList.append(dict_)
    def resume(self):
        batch = {}
        for key in self.dictList[0]:
            batch[key] = np.concatenate([data[key] for data in self.dictList])
        return batch


""" Last model in directory
    -----------------------
    Last checkpoint 
"""  
def lastModel(modelPath):
    path = glob.glob(modelPath+"/model*.pth")
    return sorted(path, key=lambda x: int(x.partition('/Model/model')[2].partition('.')[0]))[-1]
    

""" Model List in directory
    -----------------------
    Checkpoint list
"""  
def modelList(modelPath):
    path = glob.glob(modelPath+"/model*.pth")
    return sorted(path, key=lambda x: int(x.partition('/Model/model')[2].partition('.')[0]))


""" Load values (csv) to save
    -------------------------
    Load csv data
"""  
def loadValuesToSave(path):
    # Read
    df = pd.read_csv(path)
    # Delete unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return[x for x in df[:].values]
    

def getDictValue(dictionary,key):
    if key in dictionary:
        return dictionary[key]
    else:
        return None


""" Random intenger
    ---------------
    Range min, max
"""  
def randint(_min,_max):
    _range = _max - _min
    return int(_min + _range*np.random.rand())
    
