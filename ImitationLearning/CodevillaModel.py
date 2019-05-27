import glob
import h5py
import numpy  as np
import imgaug as ia
from imgaug import augmenters as iaa

from ImitationLearning.network.CodevillaNet import Codevilla19Net
from ImitationLearning.preprocessing        import dataGenerator
from ImitationLearning.config               import Config

"""
Codevilla 2019 Network
----------------------
Ref: 
    https://arxiv.org/pdf/1710.02410.pdf
"""
class CodevillaModel(object):
    def __init__(self):
        # Configure
        self._config = Config
        self.    net = None
        
        # Paths
        trainPath = self._config.trainPath
        validPath = self._config.validPath

        self._trainFile = glob.glob(trainPath + '*.h5')
        self._validFile = glob.glob(validPath + '*.h5')

        # Data Augmentation
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
        # Nets
        self.net = Codevilla19Net(self._config)

    def _commandCode(self,command):
        followLane = np.array([False,False,False])
        left       = np.array([False,False,False])
        right      = np.array([False,False,False])
        straight   = np.array([False,False,False])
        chosenOne  = np.array([ True, True, True])

        if  (command == 2): followLane = chosenOne
        elif(command == 3): left       = chosenOne
        elif(command == 4): right      = chosenOne
        elif(command == 5): straight   = chosenOne

        return (followLane, left, right, straight)

    def _batch_generator(self,file_names, batch_size = 6, masks = None):  

        batch_x = []   
        batch_y = []
        batch_s = []
        
        while True:
            for i in range(batch_size - 1):
                file_idx   = np.random.randint(len(file_names) - 1)
                sample_idx = np.random.randint(200-1)

                data = h5py.File(file_names[file_idx], 'r')

                for mask in masks:
                    if data['targets'][sample_idx][24] == mask:
                        img = self._seq.augment_image(data['rgb'][sample_idx])
                        
                        (followLane, left, right, straight) = self._commandCode(data['targets'][sample_idx][24])
                        

                        batch_x.append(self._seq.augment_image(data['rgb'][sample_idx]))
                        batch_y.append(data['targets'][sample_idx][:3])
                        batch_s.append(data['targets'][sample_idx][10]) # speed
                        
                data.close()
                
            yield ([np.array(batch_x), np.array(batch_s)], [np.array(batch_s) if mask == 1 else np.array(batch_y) for mask in masks ])

    def build(self):
        self.net.build()

    def train(self):
        self.net.fit( self._config.trainPath,
                      self._config.validPath )
    
