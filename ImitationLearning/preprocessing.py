import matplotlib
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
import random
from os      import listdir
from os.path import isfile, join

from ImitationLearning.config import Config

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
    def __init__(self, filepath = None):
        self._d = None

        self._CommandList  = ["Follow lane","Left","Right","Straight"]        
        self._Measurements = ["Steer","Gas","Brake","Speed", 
                              "Collision Other", 
                              "Collision Pedestrian",
                              "Collision Car",
                              "Opposite Lane Intersection",
                              "Sidewalk Intersection"]
        self._columns = [0,1,2,10,11,12,13,14,15]

        if filepath is not None:
            self.load(filepath)

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

    # Load
    # ....
    def load(self, filepath):
        if filepath is not None:
            self._d.close()
        self._d = h5py.File(filepath, 'r')

    # Close
    # .....
    def close(self):
        self._d.close()

    # Get Data Frames
    # ...............
    def getDataFrames(self):
        numCommandList = list(np.unique( self._d['targets'].value[:,24] ))
        DataFrames = {}

        for n in numCommandList:
            rows = (self._d['targets'].value[:,24] == n)
            data =  self._d['targets'].value[rows,self._columns]

            DataFrames[ self._CommandList[n-2] ] = pd.DataFrame(np.array(data=data,
                                                                         columns=self._Measurements))
        return DataFrames

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



"""
Data Generator
--------------
"""
class dataGenerator(object):
    def __init__(self, trainPath, valPath):
        # Paths
        self._trainFileList = [trainPath + "/" + f for f in listdir(trainPath) if isfile(join(trainPath, f))]
        self._validFileList = [  valPath + "/" + f for f in listdir(  valPath) if isfile(join(  valPath, f))]

        random.shuffle(self._trainFileList)
        random.shuffle(self._validFileList)

        # Number of files
        self._n_train = len(self._trainFileList)
        self._n_valid = len(self._validFileList)
        
        self._groupTrain = 0
        self._groupValid = 0

        self._isCompleteTrain = False
        self._isCompleteValid = False

        self._config = Config()
    

    def reset(self):
        random.shuffle(self._trainFileList)
        random.shuffle(self._validFileList)
        self._groupTrain = 0
        self._groupValid = 0

    def isEnd(self):
        return  self._isCompleteTrain and self._isCompleteValid


    def next(self):
        # Extremes
        infTrain =  self._groupTrain     *self._config.n_filesPerGroup
        supTrain = (self._groupTrain + 1)*self._config.n_filesPerGroup

        infValid =  self._groupValid     *self._config.n_filesPerGroup
        supValid = (self._groupValid + 1)*self._config.n_filesPerGroup

        # Update
        self._groupTrain = self._groupTrain + 1
        self._groupValid = self._groupValid + 1

        # Is the end
        if infTrain >= self._n_train and infValid>= self._n_valid:
            return None

        # Upper-end correction
        if supTrain > self._n_train: supTrain = self._n_train
        if supValid > self._n_valid: supValid = self._n_valid
        
        # Getting train data
        train = {}
        for path in self._trainFileList[infTrain:supTrain]:
            file = fileH5py(path)
            data = file.getDataFrames()
            for command in data.keys():
                if command in train:
                    train[command].append( data[command] )
                else:
                    train[command] = data[command]
            file.close()

        # Getting validation data
        valid = {}
        for path in self._validFileList[infValid:supValid]:
            file = fileH5py(path)
            data = file.getDataFrames()
            for command in data.keys():
                if command in valid:
                    valid[command].append( data[command] )
                else:
                    valid[command] = data[command]
            file.close()

        # Rewind
        if infTrain >= self._n_train: 
            self._groupTrain = 0
            self._isCompleteTrain = True
            random.shuffle(self._trainFileList)
        if infValid >= self._n_valid: 
            self._groupValid = 0
            self._isCompleteValid = True
            random.shuffle(self._validFileList)

        return train,valid


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
