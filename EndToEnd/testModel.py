from keras.models import load_model
import sys
import numpy as np
import glob
import os

if ('/home/victor/Documentos/Tesis/AirSim/PythonClient/' not in sys.path):
    sys.path.insert(0, '/home/victor/Documentos/Tesis/AirSim/PythonClient/')
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('/media/victor/Datos/Tesis/SelfDrivingCar/EndToEnd/model/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))

"""
Next, we'll load the model and connect to AirSim Simulator in the Landscape environment. Please ensure that the simulator is running in a different process before kicking this step off.
"""

model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')


