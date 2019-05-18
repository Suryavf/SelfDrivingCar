import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2 as cv

"""
File H5py
---------
 0. Steer,                float               15. Sidewalk Intersect,   float
 1. Gas,                  float               16. Acceleration X,       float
 2. Brake,                float               17. Acceleration Y,       float
 3. Hand Brake,           boolean             18. Acceleration Z,       float
 4. Reverse Gear,         boolean             19. Platform time,        float
 
 5. Steer Noise,          float               20. Game Time,            float
 6. Gas Noise,            float               21. Orientation X,        float
 7. Brake Noise,          float               22. Orientation Y,        float
 8. Position X,           float               23. Orientation Z,        float
 9. Position Y,           float               24. High level command    int

10. Speed,                float               25. Noise,                boolean
11. Collision Other,      float               26. Camera 
12. Collision Pedestrian, float                   (Which camera was used)
13. Collision Car,        float               27. Angle 
14. Opposite Lane Inter,  float                  (The yaw angle for this camera)

    High level command:
        - 2: Follow lane      - 4: Right
        - 3: Left             - 5: Straight
"""
class fileH5py(object):
    def __init__(self, filepath):
        self._d = h5py.File(filepath, 'r')
    
    def _getTargetsValue(self,key,index=None):
        if index is None:
            return self._d['targets'].value[  :  ,key]
        else:
            return self._d['targets'].value[index,key]
    
    def _getRGBvalue(self,index=None):
        if index is None:
            return self._d['rgb'].value
        else:
            return self._d['rgb'].value[index,:,:,:]

    # Frame
    # .....
    def frame(self,index=None):
        return self._getRGBvalue(index=index)
    
    # Steer
    # .....
    def steer(self,index=None):
        return self._getTargetsValue(0,index=index)
    
    # Gas
    # ...
    def gas(self,index=None):
        return self._getTargetsValue(1,index=index)
    
    # Brake
    # .....
    def brake(self,index=None):
        return self._getTargetsValue(2,index=index)
    
    # Hand Brake
    # ..........
    def handBrake(self,index=None):
        return self._getTargetsValue(3,index=index)
    
    # Reverse Gear
    # ............
    def reverseGear(self,index=None):
        return self._getTargetsValue(4,index=index)
    
    # Speed
    # .....
    def speed(self,index=None):
        return self._getTargetsValue(10,index=index)
    
    # Command
    # .......
    def command(self,index=None):
        return self._getTargetsValue(24,index=index)
    
    # Collision Other
    # ...............
    def collisionOther(self,index=None):
        return self._getTargetsValue(11,index=index)
    
    # Collision Pedestrian
    # ....................
    def collisionPedestrian(self,index=None):
        return self._getTargetsValue(12,index=index)
    
    # Collision Car
    # .............
    def collisionCar(self,index=None):
        return self._getTargetsValue(13,index=index)
    
    # Opposite Lane Intersection
    # ..........................
    def oppositeLaneIntersection(self,index=None):
        return self._getTargetsValue(14,index=index)
    
    # Sidewalk Intersection
    # .....................
    def sidewalkIntersection(self,index=None):
        return self._getTargetsValue(15,index=index)



from os      import listdir
from os.path import isfile, join

def dataGenerator(object):
    def __init__(self, trainpath, testpath):
        self._trainpath = trainpath
        self. _testpath =  testpath

        self._trainFileList = [trainpath + "/" + f for f in listdir(trainpath) if isfile(join(trainpath, f))]
        self. _testFileList = [ testpath + "/" + f for f in listdir( testpath) if isfile(join( testpath, f))]