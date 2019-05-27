
import random
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from os      import listdir
from os.path import isfile, join
from ImitationLearning.preprocessing import fileH5py
from ImitationLearning.config        import Config

def commandCode(command):
    followLane = np.array([False,False,False])
    left       = np.array([False,False,False])
    right      = np.array([False,False,False])
    straight   = np.array([False,False,False])
    chosenOne  = np.array([ True, True, True])

    if   command == 2: followLane = chosenOne
    elif command == 3: left       = chosenOne
    elif command == 4: right      = chosenOne
    elif command == 5: straight   = chosenOne

    return (followLane, left, right, straight)


def CoRL2017(path):
    # Paths
    fileList = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(fileList)

    config = Config()

    n_filesGroup = len(fileList)
    n_groups     = np.floor(n_filesGroup/config.filesPerGroup) - 1

    while True:
        # Groups
        for n in range(n_groups):
            
            # Files in group
            for p in fileList[n*config.filesPerGroup:(n+1)*config.filesPerGroup]:
                # Data
                file = fileH5py(p)
                frames = file.frame()
                meta   = file.getDataFrames()

                for frame,info in zip(frames,meta):
                    selectBranch = commandCode(info["Command"])

                    yield ({'input_1': frame, 
                            'input_2': info["Speed"], 
                            'input_3': selectBranch[0], 
                            'input_4': selectBranch[1], 
                            'input_5': selectBranch[2], 
                            'input_6': selectBranch[3]}, 
                           {'output' : np.array( [info["Steer"],
                                                  info["Gas"],
                                                  info["Brake"],
                                                  info["Speed"]] )})
                

