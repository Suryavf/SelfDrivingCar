import os
import numpy as np 
import tensorflow as tf
import cv2


from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping


class AirNet():

    def __init__(self,config):
        self.input = None
        self.net   = None

        self.config = config


    def SetupCallback(self):
        # ReduceLrOnPlateau 
        # -----------------
        # If the model is near a minimum and the learning rate is too high, then 
        # the model will circle around that minimum without ever reaching it. 
        # This callback will allow us to reduce the learning rate when the validation 
        # loss stops improving, allowing us to reach the optimal point.
        plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)

        # CsvLogger 
        # ---------
        # This lets us log the output of the model after each epoch, which will allow 
        # us to track the progress without needing to use the console.
        csv_callback = CSVLogger(os.path.join(self.config.dir_model, 
                                              'training_log.csv'))
        
        # ModelCheckpoint
        # --------------- 
        # Generally, we will want to use the model that has the lowest loss on the 
        # validation set. This callback will save the model each time the validation 
        # loss improves.
        checkpoint_filepath = os.path.join(self.config.dir_model, 
                                           'models', 
                                           '{0}_{1}-{2}.h5'.format('model', '{epoch:03d}', '{val_loss:.7f}'))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)

        # EarlyStopping
        # -------------
        # We will want to stop training when the validation loss stops improving. 
        # Otherwise, we risk overfitting. This monitor will detect when the validation 
        # loss stops improving, and will stop the training process when that occurs.
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        # Create
        callbacks = [plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]#, TQDMNotebookCallback()]

        return callbacks


    def build(self):
        frame_shape = self.config.shape
        state_shape = self.config.n_state

        activation = 'relu'
        padding    = 'same'

        #Create the convolutional stacks
        picInput = Input(shape=frame_shape)

        imgStack = Conv2D(16, (3, 3), activation=activation, padding=padding, name="conv0")(picInput)
        imgStack = MaxPooling2D(pool_size=(2, 2))(imgStack)
        imgStack = Conv2D(32, (3, 3), activation=activation, padding=padding, name='conv1')(imgStack)
        imgStack = MaxPooling2D(pool_size=(2, 2))(imgStack)
        imgStack = Conv2D(32, (3, 3), activation=activation, padding=padding, name='conv2')(imgStack)
        imgStack = MaxPooling2D(pool_size=(2, 2))(imgStack)
        imgStack = Flatten(   )(imgStack)
        imgStack = Dropout(0.2)(imgStack)

        #Inject the state input
        stateInput = Input(shape=state_shape)
        merged     = concatenate([imgStack, stateInput])

        # Add a few dense layers to finish the model
        merged = Dense(64, activation=activation, name='fully0')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(10, activation=activation, name='fully1')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(1, name='output')(merged)

        # Optimizer
        adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Compile
        self.net = Model(inputs=[picInput, stateInput], outputs=merged)
        self.net.compile(optimizer=adam, loss='mse')
        self.net.summary()

        

    def train(self,trainDataGenerator,evalDataGenerator):
        # Setup Callback
        callbacks = self.SetupCallback()

        n_train    = self.config.n_train
        n_eval     = self.config.n_eval
        n_epochs   = self.config.n_epochs
        batch_size = self.config.batch_size
        history    = self.net.fit_generator(trainDataGenerator, 
                                            steps_per_epoch  = n_train//batch_size, 
                                            epochs           = n_epochs, 
                                            callbacks        = callbacks, 
                                            validation_data  = evalDataGenerator, 
                                            validation_steps = n_eval//batch_size, 
                                            verbose          = 2)

        return history

    def predict(self,value):
        return self.net.predict(value)


