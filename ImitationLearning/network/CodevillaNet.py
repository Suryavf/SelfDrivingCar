import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras import backend as K

from ImitationLearning.config import Config


"""
Codevilla 2019 Network
----------------------
Ref: 
    https://arxiv.org/pdf/1710.02410.pdf
"""
class Codevilla19Net(object):
    
    def __init__(self, config):
        # Configure
        self._config = config
        self. models = {}
        
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
        
        x = Conv2D( filters, kernelSize, 
                    strides    = stride,
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
    
    
    #
    # Build
    # .....
    # 
    #              -----------------   im  -----------------
    # in_image -> | Observation Net |-----| Pred. Speed Net |--> speed
    #              -----------------   |  -----------------
    #                                  |            ---
    #                                  |    ---    | S |----| Straight  Net |--> action
    #                                  |   | F |   | W |     
    #                                   ---| U |   | I |----| TurnLeft  Net |--> action
    #              -----------------       | L |---| T |  
    # in_speed -> | Measurement Net |------| L | m | C |----| TurnRight Net |--> action
    #              -----------------   vm  | Y |   | H |
    #                                       ---    |   |----| Follow    Net |--> action
    #                                               ---
    #                                                A
    #                                                |
    #                                           in_command
    def build(self):
        shape = self._config.imageShape
        in_image = Input( shape = shape, name = 'frame')
        in_speed = Input( shape =  (1,), name = 'speed')
        
        im = self._observationNet(in_image)
        vm = self._measurementNet(in_speed)
        
        m = concatenate([im, vm], 1)
        m = self._fully(m,512)
       
        #
        # Speed  prediction
        # -----------------
        out_speed  = self._predSpeedNet(im)                    
        
        # 
        # Action prediction
        # -----------------
        #   - 2: Follow lane    - 4: Right
        #   - 3: Left           - 5: Straight
        #
        follow    = self._followNet   (m)
        straight  = self._straightNet (m)
        turnLeft  = self._turnLeftNet (m)
        turnRight = self._turnRightNet(m)
        
        inputs  = [in_image,  in_speed]
        
        self.models['follow']    = Model(inputs=inputs, outputs=[out_speed,   follow])
        self.models['straight']  = Model(inputs=inputs, outputs=[out_speed, straight])
        self.models['turnLeft']  = Model(inputs=inputs, outputs=[out_speed, turnLeft])
        self.models['turnRight'] = Model(inputs=inputs, outputs=[out_speed,turnRight])
        
        
    def train(self,image,speed,command):
        
        pass
    
    def prediction(self):
        pass