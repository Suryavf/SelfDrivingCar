
import random
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

    # Data Augmentation
    st = lambda aug: iaa.Sometimes(0.40, aug)
    oc = lambda aug: iaa.Sometimes(0.30, aug)
    rl = lambda aug: iaa.Sometimes(0.09, aug)

    seq = iaa.Sequential([rl(iaa.GaussianBlur((0, 1.5))),                                                  # blur images with a sigma between 0 and 1.5
                             rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),     # add gaussian noise to images
                             oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),                                # randomly remove up to X% of the pixels
                             oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
                             oc(iaa.Add((-40, 40), per_channel=0.5)),                                      # adjust brightness of images (-X to Y% of original value)
                             st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),                               # adjust brightness of images (X -Y % of original value)
                             rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),                   # adjust the contrast
                         ],random_order=True)

    while True:
        # Groups
        for n in range(n_groups):
            
            # Files in group
            for p in fileList[n*config.filesPerGroup:(n+1)*config.filesPerGroup]:
                # Data
                file = fileH5py(p)
                frames = file.frame()
                meta   = file.getDataFrames()
                print("Read:",p,"\n")

                z = list(zip(frames,meta))
                random.shuffle(z)

                for frame,info in z:
                    selectBranch = commandCode(info["Command"])
                    frame        = seq.augment_image(frame)

                    yield ({'input_1': frame.astype(float)/255, 
                            'input_2': info["Speed"], 
                            'input_3': selectBranch[0], 
                            'input_4': selectBranch[1], 
                            'input_5': selectBranch[2], 
                            'input_6': selectBranch[3]}, 
                           {'output' : np.array( [info["Steer"]/1.2,
                                                  info["Gas"  ],
                                                  info["Brake"],
                                                  info["Speed"]/85] )})
                file.close()
                
