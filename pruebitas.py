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

user_maestria

laurita1!


ssh 192.168.0.41 -l user_maestria



conv1: 32x(5x5) + 2
conv2: 32x(3x3)
conv3: 64x(3x3) + 2
conv4: 64x(3x3)
conv5: 128x(3x3) + 2
conv6: 128x(3x3)
conv7: 256x(3x3)
conv8: 256x(3x3)

fully1: 512
fully1: 512


batch normalization:
after convolutional layers

dropout :
after fully-connected hidden layers: 50%
after          convolutional layers: 20%

Optimizator:
Adam
minibatches: 120
initLearningRate: 0.0002

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
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras import backend as K

class Config(object):
    
    imageShape   = (88,200)
    activation   = 'relu'
    padding      = 'same'
    convDropout  = 0.2
    fullyDropout = 0.5



"""
Codevilla 2018 Network
----------------------
Ref: 
    https://arxiv.org/pdf/1710.02410.pdf
"""
class Codevilla18Net(object):
    
    def __init__(self, config):
        # Configure
        self._config = config
        self._branchConfig = [["Steer", "Gas", "Brake"], 
                              ["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"], 
                              ["Steer", "Gas", "Brake"], 
                              ["Speed"]]
        
        self.model = None
        
        # Counts
        self._countConv       = 0
        self._countPool       = 0
        self._countBatchNorm  = 0
        self._countDropout    = 0
        self._countFully      = 0
        
    def _conv(self, x, filters, kernelSize, stride):
        self._countConv      += 1
        self._countBatchNorm += 1
        self._countDropout   += 1
        
        x = Conv2D( filters, kernelSize, stride,
                    activation = self._config.activation, 
                    padding    = self._config.   padding, 
                    name       = 'conv{}'.format(self._countConv))(x)
        x = BatchNormalization(
                    name       = 'BatchNorm{}'.format(self._countBatchNorm))(x)
        x = Dropout(self._config.convDropout,
                    name       = 'Dropout{}'  .format(self._countDropout  ))(x)
        return x
    
    def _fully(self, x, units):
        self._countFully   += 1
        self._countDropout += 1
        
        x = Dense(units, activation = self._config.activation,
                         name       = 'fully{}'  .format(self._countFully  ))(x)
        x = Dropout(self._config.fullyDropout,
                         name       = 'Dropout{}'.format(self._countDropout))(x)
        return x
    
    def _observationNet(self,x):
        
        # Convolutional stage
        x = self._conv(x,32,5,2)
        x = self._conv(x,32,3,1)
        
        x = self._conv(x,64,3,2)
        x = self._conv(x,64,3,1)
        
        x = self._conv(x,128,3,2)
        x = self._conv(x,128,3,1)
        
        x = self._conv(x,256,3,2)
        x = self._conv(x,256,3,1)
        
        x = Flatten()(x)
        
        # Fully stage
        x = self._fully(x,512)
        x = self._fully(x,512)
        
        return x
    
    def _measurementNet(self,x):
        x = self._fully(x,128)
        x = self._fully(x,128)
        return x
    
    def _controlNet(self,x):
        x = self._fully(x,256)
        x = self._fully(x,256)
        return x
    
    def _predSpeedNet(self,x):
        x = self._fully(x,256)
        x = self._fully(x,256)
        
        self._countFully += 1
        x = Dense(1, activation = self._config.activation,
                     name       = 'fully{}'.format(self._countFully))(x)
        return x
    
    def _commandNet(self,x):
        x = self._fully(x,256)
        x = self._fully(x,256)
        
        self._countFully += 1
        x = Dense(3, activation = self._config.activation,
                     name       = 'fully{}'.format(self._countFully))(x)
        return x
    
    def _straightNet(self,x):
        return self._commandNet(x)
    
    def _turnLeftNet(self,x):
        return self._commandNet(x)
    
    def _turnRightNet(self,x):
        return self._commandNet(x)
    
    def _followNet(self,x):
        return self._commandNet(x)
    
    def build(self):
        im = Input( shape = self._config.imageShape, name =  'frame')
        vm = Input( shape = (None,1), name =  'speed')
        cm = Input( shape = (None,1), name ='command')
        
        im = self._observationNet(im)
        vm = self._measurementNet(vm)
        
        m = concatenate([im, vm], 1)
        m = self._fully(m,512)
        
        # Outputs
        speed  = self._predSpeedNet(im)                     # Speed  prediction
        action = K.switch( K.equal( cm, 5),                 # Action prediction
                           self._straightNet(m), 
                           K.switch( K.equal(cm,3),
                                     self._turnLeftNet(m),
                                     K.switch( K.equal(cm,4),
                                               self._turnRightNet(m),
                                               self.   _followNet(m)
                                              )
                                    ) 
                          )
        # int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
        inputs  = [im,  vm,  cm]
        outputs = [speed,action]
        
        self.model = Model(inputs=inputs, outputs=outputs)
        pass
        
    
    
    
# --------------------------------------------------------------------------------------------    
class transformation(object):
    
    def __init__(self,image = None):
        self.image = image
    
    #
    # Gamma correction
    # ----------------
    # Ref: https://en.wikipedia.org/wiki/Gamma_correction
    def _GammaCorrection(self,gamma):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv.LUT(self.image, lookUpTable)
    
    #
    # Contrast
    # --------
    # contrast=[-255,255]
    # Ref: https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    def _Contrast(self):
        #factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        #img = factor*(self.image.astype('float') - 128) + 128
        img = self.image.copy()
        
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        
        return cdf[img]
    
    #
    # Change tone
    # -----------
    def _Tone(self,mod):
        hsv = cv.cvtColor(self.image, cv.COLOR_RGB2HSV)
        hsv[:,:,0] = hsv[:,:,0] + mod
        return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    
    def _GaussianBlur(self,order):
        return cv.GaussianBlur(self.image,(order,order),0)
    
    def _SaltPepperNoise(self,prob):
        inf =     prob/2
        sup = 1 - prob/2
        rnd = np.random.rand(self.image.shape[0], self.image.shape[1])
        
        noise = self.image.copy()
        
        noise[rnd < inf,:] =  [  0,  0,  0]
        noise[rnd > sup,:] =  [255,255,255]
        
        return noise
    
    def random(self):
        select = np.random.randint(5) + 1
        if   select == 1:
            return self._GammaCorrection(np.random.uniform(low=0.1,high=2))
        elif select == 2:
            return self._Contrast()
        elif select == 3:
            return self._Tone(np.random.uniform(low=0,high=255))
        elif select == 4:
            return self._GaussianBlur(2*np.random.randint(5)+1)
        elif select == 5:
            return self._SaltPepperNoise(np.random.uniform(low=0.0,high=0.1))

#tr = transformation(img)
#plt.imshow(tr._Contrast())
#plt.imshow(tr._GammaCorrection(0.5))

"""

inf =     prob/2
sup = 1 - prob/2
rnd = np.random.rand(img.shape[0], img.shape[1])

noisy = np.zeros(img.shape)
noisy[rnd < inf] = -10
noisy[rnd > sup] =  10
h = img + noisy.astype('uint8')



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

