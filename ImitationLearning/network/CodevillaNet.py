import os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate, Multiply, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras import backend as K

from ImitationLearning.config import Config

def loss(yTrue,yPred):
    K.mean(yTrue - yPred)


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
        self. model = None
        
        # Counts
        self._countConv       = 0
        self._countPool       = 0
        self._countBatchNorm  = 0
        self._countDropout    = 0
        self._countFully      = 0
        


    def _SetupCallback(self):
        # CsvLogger 
        # ---------
        # This lets us log the output of the model after each epoch, which will allow 
        # us to track the progress without needing to use the console.
        csv_callback = CSVLogger(os.path.join(self._config.modelPath, 
                                              'training_log.csv'))
        
        # ModelCheckpoint
        # --------------- 
        # Generally, we will want to use the model that has the lowest loss on the 
        # validation set. This callback will save the model each time the validation 
        # loss improves.
        checkpoint_filepath = os.path.join(self._config.modelPath, 
                                           'models', 
                                           '{0}_{1}-{2}.h5'.format('model', '{epoch:03d}', '{val_loss:.7f}'))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)

        # Create
        callbacks = [csv_callback, checkpoint_callback]

        return callbacks



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
        
        # Data inputs
        in_image = Input( shape = shape, name = 'frame')
        in_speed = Input( shape =  (1,), name = 'speed')

        # Conditional inputs
        in_Follow    = Input(shape = (3,), name = 'cmdFollow'   )
        in_Straight  = Input(shape = (3,), name = 'cmdStraight' )
        in_TurnLeft  = Input(shape = (3,), name = 'cmdTurnLeft' )
        in_TurnRight = Input(shape = (3,), name = 'cmdTurnRight')

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
        follow    = self._followNet   (m)
        straight  = self._straightNet (m)
        turnLeft  = self._turnLeftNet (m)
        turnRight = self._turnRightNet(m)

        follow    = Multiply()([follow   ,in_Follow   ]) 
        straight  = Multiply()([straight ,in_Straight ]) 
        turnLeft  = Multiply()([turnLeft ,in_TurnLeft ]) 
        turnRight = Multiply()([turnRight,in_TurnRight]) 
        
        out_action = Add()([follow,straight,turnLeft,turnRight])

        # Input/Output
        inputs  = [in_image,in_speed,
                   in_Follow,in_Straight,
                   in_TurnLeft,in_TurnRight]
        outputs = concatenate([out_action, out_speed], 1)
        
        # Model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        #
        # Optimizer
        # ---------
        optimizer = Adam(lrate = 0.0002, beta_1= 0.7, beta_2 = 0.85)
        self.model.compile(optimizer,loss='mean_squared_error')


    #
    # Fit model
    # .........
    def fit(self,inTrain,outTrain,
                 inValid,outValid,
                 command):

        # Setup Callback
        callbacks = self._SetupCallback()

        self.models[command].fit(inTrain, outTrain,
                                 validation_data = (inValid,outValid),
                                 batch_size      = self._config.batch_size,
                                 callbacks       = callbacks )
    

    #
    # Prediction
    # ..........
    def prediction(self,inTest,command):
        self.models[command].predict(inTest)