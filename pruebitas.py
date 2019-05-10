import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2 as cv

#  /usr/bin/python3 pruebitas.py

# Read Data
# ['rgb', 'targets']
path = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqTrain/"
raw = h5py.File(path + "data_03685.h5", 'r')

data   = raw[  'rgb'  ].value
target = raw['targets'].value
"""
 0. Steer, float
 1. Gas, float
 2. Brake, float
 3. Hand Brake, boolean
 4. Reverse Gear, boolean
 
 5. Steer Noise, float
 6. Gas Noise, float
 7. Brake Noise, float
 8. Position X, float
 9. Position Y, float

10. Speed, float
11. Collision Other, float
12. Collision Pedestrian, float
13. Collision Car, float
14. Opposite Lane Inter, float

15. Sidewalk Intersect, float
16. Acceleration X,float
17. Acceleration Y, float
18. Acceleration Z, float
19. Platform time, float

20. Game Time, float
21. Orientation X, float
22. Orientation Y, float
23. Orientation Z, float
24. High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)

25. Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) )
26. Camera (Which camera was used)
27. Angle (The yaw angle for this camera)
"""

#
# Visualization
# -------------
#
frame = 1
rgb = data[frame,:,:,:]
plt.imshow(rgb)
plt.axis('off')



#
# Network
# -------
#
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping




config.imageShape = (88,200)
config.activation = 'relu'
config.padding    = 'same'


"""
Codevilla 2018 Network
----------------------
Ref: 
    https://arxiv.org/pdf/1710.02410.pdf
"""
class CodevillaNet(object):
    
    def __init__(self, config):
        # Configure
        self._config = config
        
        # Counts
        self._countConv       = 0
        self._countPool       = 0
        self._countBatchNorm  = 0
        self._countActivation = 0
        self._countDropout    = 0
        self._countFc         = 0 # ??
        self._countLSTM       = 0
        self._countSoftmax    = 0
        
        

    def conv(self, filters, kernelSize, stride):
        self._countConv += 1
        
        return Conv2D(filters, kernelSize, stride,
                               activation = self._config.activation, 
                               padding    = self._config.   padding, 
                               name       = 'conv{}'.format(self._countConv))
        
    def build(self):
        pass
        
        
class transformation(object):
    
    def __init__(self):
        pass
    
    def _HSV(self,img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        



"""
#Create the convolutional stacks
picInput = Input(shape=(88,200))

# Image module
imit = Conv2D(32, (5, 5), strides    =     (2, 2),
                          activation = activation, 
                          padding    =    padding, name="conv1")(picInput)
imit = Conv2D(32, (3, 3), strides    =     (1, 1),
                          activation = activation, 
                          padding    =    padding, name="conv2")(imit)

imit = Conv2D(64, (3, 3), strides    =     (2, 2),
                          activation = activation, 
                          padding    =    padding, name="conv3")(imit)
imit = Conv2D(64, (3, 3), strides    =     (1, 1),
                          activation = activation, 
                          padding    =    padding, name="conv4")(imit)

imit = Conv2D(128, (3, 3), strides    =     (2, 2),
                           activation = activation, 
                           padding    =    padding, name="conv5")(imit)
imit = Conv2D(128, (3, 3), strides    =     (1, 1),
                           activation = activation, 
                           padding    =    padding, name="conv6")(imit)

imit = Conv2D(256, (3, 3), strides    =     (1, 1),
                           activation = activation, 
                           padding    =    padding, name="conv7")(imit)
imit = Conv2D(256, (3, 3), strides    =     (1, 1),
                           activation = activation, 
                           padding    =    padding, name="conv8")(imit)



imgStack = MaxPooling2D(pool_size=(2, 2))(imgStack)



imgStack = Conv2D(32, (3, 3), activation=activation, padding=padding, name='conv1')(imgStack)
imgStack = MaxPooling2D(pool_size=(2, 2))(imgStack)
imgStack = Conv2D(32, (3, 3), activation=activation, padding=padding, name='conv2')(imgStack)
imgStack = MaxPooling2D(pool_size=(2, 2))(imgStack)
imgStack = Flatten(   )(imgStack)
imgStack = Dropout(0.2)(imgStack)
"""

