from config import Config
from config import Global

import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import datetime
import os

# Settings
_global = Global()
_config = Config()


""" Number of iterations to data train time
    ---------------------------------------
    Args:
        ite: Iteration number

    Return: time in text
"""
def iter2time(ite):
    time = ite*_config.batch_size/_global.framePerSecond

    # Hours
    hour = np.floor(time/3600)
    time = time - hour*3600

    # Minutes
    minute = np.floor(time/60)
    time = time - minute*60

    # Seconds
    second = time

    # Text
    txt = ""
    if(  hour>0): txt = txt + str( hour ) + "h\t"
    else        : txt = txt + "\t"

    if(minute>0): txt = txt + str(minute) + "m\t"
    else        : txt = txt + "\t"

    return txt + str(second) + "s"


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

""" Check directory
    ---------------
    Args:
        d: directory by check

"""
def checkdirectory(d):
    if not os.path.exists(d):
        os.makedirs(d)
    
