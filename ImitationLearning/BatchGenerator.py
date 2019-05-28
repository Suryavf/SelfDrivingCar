
import random
import numpy as np
import keras
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


"""
CoRL2017 Data Generator
-----------------------
"""
class CoRL2017(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path):
        
        # Config
        self._config = Config()

        'Initialization'
        self._batch_size      = self._config.batch_size
        self._fileList        = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
        self._n_filesGroup    = len(self._fileList)
        self._n_groups        = np.floor(self._n_filesGroup/self._config.filesPerGroup) - 1
        self._steps_per_epoch = self._config.steps_per_epoch

        'Image augmentation'
        st = lambda aug: iaa.Sometimes(0.40, aug)
        oc = lambda aug: iaa.Sometimes(0.30, aug)
        rl = lambda aug: iaa.Sometimes(0.09, aug)

        self._seq = iaa.Sequential([rl(iaa.GaussianBlur((0, 1.5))),                                               # blur images with a sigma between 0 and 1.5
                                    rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),     # add gaussian noise to images
                                    oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),                                # randomly remove up to X% of the pixels
                                    oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
                                    oc(iaa.Add((-40, 40), per_channel=0.5)),                                      # adjust brightness of images (-X to Y% of original value)
                                    st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),                               # adjust brightness of images (X -Y % of original value)
                                    rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),                   # adjust the contrast
                                  ],random_order=True)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        fileBatch = self._fileList[index*self._config.filesPerGroup:(index+1)*self._config.filesPerGroup]
        
        # Generate data
        inputs, output = self.__data_generation(fileBatch)

        return inputs, output

    def on_epoch_end(self):
        random.shuffle(self._fileList)

    def __data_generation(self, fileBatch):
        'Initialize'
        Frames    = list()  # [H,W,C] float
        Speed     = list()  # [1]     float
        Follow    = list()  # [3]     boolean
        Straight  = list()  # [3]     boolean
        TurnLeft  = list()  # [3]     boolean
        TurnRight = list()  # [3]     boolean

        Outputs   = list()  # [4]     float

        for p in fileBatch:
            # Data
            file   = fileH5py(p)

            # Inputs
            Frames   .append( file.       frame() )
            Speed    .append( file.       speed() )
            Follow   .append( file.   getFollow() )
            Straight .append( file. getStraight() )
            TurnLeft .append( file. getTurnLeft() )
            TurnRight.append( file.getTurnRight() )

            # Outputs
            Outputs  .append( file.getActionSpeed() )

            file.close()

        # List to np.array
        Frames    = np.concatenate(Frames   )
        Speed     = np.concatenate(Speed    )
        Follow    = np.concatenate(Follow   )
        Straight  = np.concatenate(Straight )
        TurnLeft  = np.concatenate(TurnLeft )
        TurnRight = np.concatenate(TurnRight)
        Outputs   = np.concatenate(Outputs  )

        # Random index
        index = np.array(range( Frames.shape[0] ))
        np.random.shuffle(index)
        Frames    = Frames   [index]
        Speed     = Speed    [index]
        Follow    = Follow   [index]
        Straight  = Straight [index]
        TurnLeft  = TurnLeft [index]
        TurnRight = TurnRight[index]
        Outputs   = Outputs  [index]

        return [Frames,Speed,Follow,Straight,TurnLeft,TurnRight], Outputs

