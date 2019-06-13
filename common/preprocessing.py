import matplotlib
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
import random
import os
from os      import listdir
from os.path import isfile, join

from tqdm import tqdm

from config import Config

FRAMES_PER_FILE = 200
FILES_PER_GROUP = 100


"""
File H5py
---------
targets:

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
---------------------------------------------------------------
Train
===============================================================
 0. Steer,       (-1.0845270156860352, 1.1989188194274902)
 1. Gas,         (0.0, 1.0)
 2. Brake,       (0.0, 1.0)
 5. Steer Noise, (-1.0845270156860352, 1.0100972652435303)
 6. Gas Noise,   (0.0, 1.0)
 7. Brake Noise, (0.0, 1.0)
10. Speed,       (-18.73902702331543, 82.63579559326172)
---------------------------------------------------------------
Test
===============================================================
 0. Steer,       (-1.0, 1.1992229223251343)
 1. Gas,         ( 0.0, 1.0)
 2. Brake,       ( 0.0, 1.0)
 5. Steer Noise, (-1.0, 1.0)
 6. Gas Noise,   ( 0.0, 1.0)
 7. Brake Noise, ( 0.0, 1.0)
10. Speed,       (-15.157238960266113, 82.66552734375)

Gas   = Gas   Noise
Brake = Brake Noise
"""
class fileH5py(object):
    def __init__(self, filepath = None):
        self._d        = None
        self._n_frames = 0
        
        self._CommandList  = ["Follow lane","Left","Right","Straight"]        
        self._Measurements = ["Steer","Gas","Brake","Speed","Command"]#, 
                              #"Collision Other", 
                              #"Collision Pedestrian",
                              #"Collision Car",
                              #"Opposite Lane Intersection",
                              #"Sidewalk Intersection"]
        self._columns = [0,1,2,10,24]#,11,12,13,14,15]

        if filepath is not None:
            self.load(filepath)

    def _getTargetsValue(self,key,index=None):
        if index is None:
            return self._d['targets'][  :  ,key]
        else:
            return self._d['targets'][index,key]
    
    def _getRGBvalue(self,index=None):
        if index is None:
            return self._d['rgb'][index]
        else:
            return self._d['rgb'][:]

    # Load
    # ....
    def load(self, filepath):
        if self._d is not None:
            self._d.close()
        self._d        = h5py.File(filepath, 'r')
        self._n_frames = self._d['targets'].value.shape[0]
        
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

    # Get Output:
    # ................................
    #   0. Steer,                float         5. Steer Noise
    #   1. Gas,                  float 
    #   2. Brake,                float 
    #  10. Speed,                float 
    def output(self):
        indices = [0,1,2,10]
        actionSpeed = self._d['targets'][:,indices]
        actionSpeed[:,0] = actionSpeed[:,0]/1.2 # Steer
        actionSpeed[:,3] = actionSpeed[:,3]/ 85 # Speed
        return actionSpeed
    
    # Get Output:
    # ................................
    #   5. Steer Noise,          float
    #   1. Gas,                  float 
    #   2. Brake,                float 
    #  10. Speed,                float 
    def outputNoise(self):
        indices = [5,1,2,10]
        actionSpeed = self._d['targets'][:,indices]
        actionSpeed[:,0] = actionSpeed[:,0]/1.2 # Steer
        actionSpeed[:,3] = actionSpeed[:,3]/ 85 # Speed
        return actionSpeed

    # Follow (2)
    # ..........
    def getFollow(self):
        template = np.array([[ True, True, True]])
        isfollow = (self.command() == 2).reshape(self._n_frames,1)
        return isfollow*template

    # Turn Left (3)
    # .............
    def getTurnLeft(self):
        template = np.array([ True, True, True])
        isTurnLeft = (self.command() == 3).reshape(self._n_frames,1)
        return isTurnLeft*template

    # Turn Right(4)
    # .............
    def getTurnRight(self):
        template = np.array([ True, True, True])
        isTurnRight = (self.command() == 4).reshape(self._n_frames,1)
        return isTurnRight*template

    # Straight (5)
    # ............
    def getStraight(self):
        template = np.array([ True, True, True])
        isStraight = (self.command() == 5).reshape(self._n_frames,1)
        return isStraight*template

    # Frame
    # .....
    def frame(self,index=None):
        img = self._getRGBvalue(index=index)
        return img.astype(float)/255
    
    # Steer
    # .....
    def steer(self,index=None):
        return self._getTargetsValue(0,index=index)/1.2
    def steerNoise(self,index=None):
        return self._getTargetsValue(5,index=index)/1.2

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
        return self._getTargetsValue(10,index=index)/85

    # Command
    # .......
    #   - 2: Follow lane      - 4: Right
    #   - 3: Left             - 5: Straight
    def command(self,index=None):
        return self._getTargetsValue(24,index=index)
    def commandOneHot(self,index=None):
        commad = self._getTargetsValue(24,index=index) - 2
        commad = commad.astype(int)
        OneHot = np.zeros((FRAMES_PER_FILE,4))
        OneHot[ np.arange(FRAMES_PER_FILE),commad ] = 1
        return OneHot
    
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
Cooking dataset
---------------
"""
class cooking(object):
    def __init__(self):
        # Load config
        self._config    = Config()
        self._validPath = self._config.validPath
        self._trainPath = self._config.trainPath

        # Create directory
        if not os.path.exists(self._config.modelPath):
            os.makedirs(self._config.modelPath)
        if not os.path.exists(self._config.graphPath):
            os.makedirs(self._config.graphPath)

        # File List
        self._validFileList = [self._validPath + "/" + f for f in listdir(self._validPath) if isfile(join(self._validPath, f))]

        fileList = [self._trainPath + "/" + f for f in listdir(self._trainPath) if isfile(join(self._trainPath, f))]
        self._trainFileList1 = fileList.copy()
        self._trainFileList2 = fileList.copy()

        random.shuffle(self._trainFileList1)
        random.shuffle(self._trainFileList2)


    #
    # Training files
    # ..............
    def _trainingFiles(self):
        n_train = len(self._trainFileList1)
        n = 0

        print("\n")
        print("Training files")
        print("..............")

        # Initialize
        frame   = list()
        speed   = list()
        output  = list()
        command = list()

        for i in tqdm(range(n_train)):
            # Real control
            file1 = fileH5py(self._trainFileList1[i])
            frame  .append( file1.frame        () )
            speed  .append( file1.speed        () )
            output .append( file1.commandOneHot() )
            command.append( file1.output       () )
            file1.close()

            # Noise control
            file2 = fileH5py(self._trainFileList2[i])
            frame  .append( file2.frame        () )
            speed  .append( file2.speed        () )
            output .append( file2.commandOneHot() )
            command.append( file2.output       () )
            file2.close()

            # Save file
            if i%(FILES_PER_GROUP/2) == 0 or i==(n_train-1):
                # List to np.array
                frame   = np.concatenate(   frame )
                speed   = np.concatenate(   speed )
                output  = np.concatenate(  output )
                command = np.concatenate( command )
                
                # File name
                filename  = "Train_" + str(n) + "_" + str(frame.shape[0]) + ".h5"
                
                # To H5py
                with h5py.File(filename, 'w') as hf:
                    hf.create_dataset(  "frame", data=frame   )
                    hf.create_dataset(  "speed", data=speed   )
                    hf.create_dataset( "output", data=output  )
                    hf.create_dataset("command", data=command )

                # Initialize
                frame   = list()
                speed   = list()
                command = list()
                output  = list()

                # Update n
                n = n + 1
            

    #
    # Validation files
    # ................
    def _validationFiles(self):
        n_valid = len(self._validFileList)
        n = 0

        # Initialize
        frame   = list()
        speed   = list()
        output  = list()
        command = list()

        for i in tqdm(range(n_valid)):
            # Real control
            file = fileH5py(self._validFileList[i])
            frame  .append( file.frame        () )
            speed  .append( file.speed        () )
            output .append( file.commandOneHot() )
            command.append( file.output       () )
            file.close()

            # Save file
            if i%(FILES_PER_GROUP) == 0 or i==(n_valid-1):
                # List to np.array
                frame   = np.concatenate(   frame )
                speed   = np.concatenate(   speed )
                output  = np.concatenate(  output )
                command = np.concatenate( command )
                
                # File name
                filename  = "Valid_" + str(n) + "_" + str(frame.shape[0]) + ".h5"
                
                # To H5py
                with h5py.File(filename, 'w') as hf:
                    hf.create_dataset(  "frame", data=frame   )
                    hf.create_dataset(  "speed", data=speed   )
                    hf.create_dataset( "output", data=output  )
                    hf.create_dataset("command", data=command )

                # Initialize
                frame   = list()
                speed   = list()
                command = list()
                output  = list()

                # Update n
                n = n + 1
            
    def run(self):
        
        # Read training files
        self._trainingFiles()

        # Read validation files
        self._validationFiles()


"""
Transformation
--------------
"""
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