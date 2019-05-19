#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:19:29 2019

@author: victor
"""

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks  import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras import backend as K

class Config(object):
    
    imageShape   = (88,200,3)
    activation   = 'relu'
    padding      = 'same'
    convDropout  = 0.2
    fullyDropout = 0.5


_config = Config()
im = Input( shape = _config.imageShape, name =  'frame')
x = Conv2D( 32, (5,5), strides = 2,
	    activation = _config.activation, 
	    padding    = _config.padding)(im)




def _conv(x, filters, kernelSize, stride):
	x = Conv2D( filters, (kernelSize, kernelSize),
                    strides    = stride,
		    activation = _config.activation, 
		    padding    = _config.   padding)(x)
	x = BatchNormalization()(x)
	x = Dropout(_config.convDropout)(x)
	return x

def _fully(x, units):
	x = Dense(units, activation = _config.activation)(x)
	x = Dropout(_config.fullyDropout)(x)
	return x

def _observationNet(x):
	# Convolutional stage
	x = _conv(x,32,5,2)
	x = _conv(x,32,3,1)

	x = _conv(x,64,3,2)
	x = _conv(x,64,3,1)

	x = _conv(x,128,3,2)
	x = _conv(x,128,3,1)

	x = _conv(x,256,3,2)
	x = _conv(x,256,3,1)

	x = Flatten()(x)

	# Fully stage
	x = _fully(x,512)
	x = _fully(x,512)

	return x

def _measurementNet(x):
	x = _fully(x,128)
	x = _fully(x,128)
	return x

def _controlNet(x):
	x = _fully(x,256)
	x = _fully(x,256)
	return x

def _predSpeedNet(x):
	x = _fully(x,256)
	x = _fully(x,256)
	x = Dense(1, activation = _config.activation)(x)
	return x

def _commandNet(x):
	x = _fully(x,256)
	x = _fully(x,256)
	x = Dense(3, activation = _config.activation)(x)
	return x

def _straightNet(x):
	return _commandNet(x)

def _turnLeftNet(x):
	return _commandNet(x)

def _turnRightNet(x):
	return _commandNet(x)

def _followNet(x):
	return _commandNet(x)



im = Input( shape = _config.imageShape, name =  'frame')
vm = Input( shape = (None,1), name =  'speed')
cm = Input( shape = (None,1), name ='command')

im = _observationNet(im)
vm = _measurementNet(vm)

m = concatenate([im, vm], 1)
m = _fully(m,512)

speed  = _predSpeedNet(im)