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

        # Nets
        self.net = Codevilla19Net(self._config)

    def build(self):
        self.net.build()

    def train(self):
        self.net.fit( self._config.trainPath,
                      self._config.validPath )
    
