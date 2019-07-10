from config import Config
from config import Global
import numpy as np

# Settings
__global = Global()
__config = Config()


""" Number of iterations to data train time
    ---------------------------------------
    Args:
        ite: Iteration number

    Return: time in text
"""
def iter2time(ite):
    time = ite*__config.batch_size/__global.framePerSecond

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