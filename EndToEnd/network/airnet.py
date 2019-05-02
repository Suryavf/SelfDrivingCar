import numpy as np 
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam

class AirNet():

    def __init__(self,config):
        self.input = None
        self.net   = None

        self.config = config

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
        merged = Dense(10, activation=activation, name='fully2')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(1, name='output')(merged)

        # Optimizer
        adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Compile
        self.net = Model(inputs=[picInput, stateInput], outputs=merged)
        self.net.compile(optimizer=adam, loss='mse')

        self.net.summary()
