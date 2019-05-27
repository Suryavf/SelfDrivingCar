import os
import tensorflow as tf
import math
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate, Multiply, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import backend as K

from ImitationLearning.BatchGenerator import CoRL2017 as BatchGenerator
from ImitationLearning.config         import Config

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
        

    def _step_decay(self,epoch):
        initial_lrate = self._config.learning_rate_initial
        drop          = self._config.learning_rate_decay_factor
        epochs_drop   = self._config.learning_rate_decay_steps
        
        return initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    def _loss(self,y_true,y_pred):
        l1 = self._config.lambda_steer * K.mean( K.abs( y_true[0] - y_pred[0] ) )
        l2 = self._config.lambda_gas   * K.mean( K.abs( y_true[1] - y_pred[1] ) )
        l3 = self._config.lambda_brake * K.mean( K.abs( y_true[2] - y_pred[2] ) )
        l4 = self._config.lambda_speed * K.mean( K.abs( y_true[3] - y_pred[3] ) )

        return l1 + l2 + l3 + l4

    def _mseSteer(self,y_true,y_pred):
        return math.sqrt( K.mean(K.pow( y_true[0]-y_pred[0] ,2)) )

    def _mseGas(self,y_true,y_pred):
        return math.sqrt( K.mean(K.pow( y_true[1]-y_pred[1] ,2)) )

    def _mseBrake(self,y_true,y_pred):
        return math.sqrt( K.mean(K.pow( y_true[2]-y_pred[2] ,2)) )

    def _mseSpeed(self,y_true,y_pred):
        return math.sqrt( K.mean(K.pow( y_true[3]-y_pred[3] ,2)) )

    def _SetupCallback(self):

        # Learning Rate Schedules 
        # -----------------------
        # Ref: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
        lrate = LearningRateScheduler(self._step_decay)

        # TensorBoard 
        # -----------
        tbCallBack = TensorBoard(log_dir = self._config.graphPath, 
                                 histogram_freq = 0, 
                                 write_graph=True, write_images=True)
        
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
                                           'models','{0}_{1}-{2}.h5'.format('model', '{epoch:03d}', '{val_loss:.7f}'))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1, period=self._config.epoch_per_save)

        # Create
        callbacks = [lrate, csv_callback, checkpoint_callback, tbCallBack]

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
        optimizer = Adam(lrate  = self._config.adam_lrate, 
                         beta_1 = self._config.adam_beta_1, 
                         beta_2 = self._config.adam_beta_2)
        self.model.compile( optimizer,
                            loss    = self._loss,
                            metrics = ['mse',
                                       self._mseSteer,
                                       self._mseGas,
                                       self._mseBrake])

    #
    # Fit model
    # .........
    def fit(self,trainPath,validPath):

        # Setup Callback
        callbacks = self._SetupCallback()
        
        self.model.fit_generator(   BatchGenerator(trainPath),
                                    validation_data = BatchGenerator(validPath),
                                    steps_per_epoch = self._config.steps_per_epoch,
                                    epochs          = self._config.epochs,
                                    callbacks       = callbacks )

    #
    # Load model
    # ..........
    def load(self,modelPath):
        self.model.load_weights(modelPath)

    #
    # Prediction
    # ..........
    def prediction(self,inTest):
        self.model.predict(inTest)
    
