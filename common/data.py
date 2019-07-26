import glob
import random
import numpy as np
import torch
import h5py
from   torch.utils.data import Dataset
from   torchvision      import transforms
from   imgaug           import augmenters as iaa


FRAMES_PER_FILE = 200

class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)

""" Carla Dataset
    -------------
    Data generator for Carla Dataset.
    Ref: https://github.com/onlytailei/carla_cil_pytorch/blob/uncertain_open/carla_loader.py
        * Input: path       (str)
                 train      (bool)
                 branches   (bool)
                 multimodal (bool)
                 speedReg   (bool)
        * Output: action (vector: 3) [Steer,Gas,Brake]

    Methods:
        @forward: Forward network
            - img: image input
            - vm : speed input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model

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

"""
class CoRL2017Dataset(Dataset):
    def __init__(self, path, train      = True,
                             branches   = False,
                             multimodal = False,
                             speedReg   = False):
        self._files = glob.glob(path+'*.h5')
        self._files.sort()
        self._transform = None

        self._isTrain      =      train
        self._isBranches   =   branches
        self._includeSpeed = multimodal or speedReg
        
        self.build()

    def build(self):
        if self._isTrain:
            self._transform = transforms.Compose([#transforms.ToTensor()])#
                transforms.RandomOrder([
                    RandomTransWrapper( seq=iaa.GaussianBlur((0, 1.5)),
                                        p=0.09),
                    RandomTransWrapper( seq=iaa.AdditiveGaussianNoise(loc=0,scale=(0.0, 0.05),per_channel=0.5),
                                        p=0.09),
                    RandomTransWrapper( seq=iaa.Dropout((0.0, 0.10), per_channel=0.5),
                                        p=0.3),
                    RandomTransWrapper( seq=iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5),
                                        p=0.3),
                    RandomTransWrapper( seq=iaa.Add((-20, 20), per_channel=0.5),
                                        p=0.3),
                    RandomTransWrapper( seq=iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                        p=0.4),
                    RandomTransWrapper( seq=iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),
                                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self._transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return FRAMES_PER_FILE * len(self._files)

    def trainingRoutine(self,img,target):
        # Steering angle
        target[0] = target[0]/1.2   # Angle (max 1.2 rad)
            
        if self._isBranches:
            # Control output
            command = int(target[24])-2
            out = np.zeros((4, 3), dtype=np.float32)  # modes x actions (controls)
            out[command,:] = target[:3]

            # Mask
            mask = np.zeros((4, 3), dtype=np.float32)
            mask[command,:] = 1

            if self._includeSpeed:
                # Speed input/output (max 90km/h)
                speed = np.array([target[10]/90,]).astype(np.float32)
                return img, speed, out.reshape(-1), mask.reshape(-1)
            else:
                return img, out.reshape(-1), mask.reshape(-1)

        else:
            out = target[:3]

            if self._includeSpeed:
                # Speed input/output (max 90km/h)
                speed = np.array([target[10]/90,]).astype(np.float32)
                return img, speed, out.reshape(-1)
            else:
                return img, out.reshape(-1)

    def evaluationRoutine(self,img,target):
        # Steering angle
        target[0] = target[0]/1.2   # Angle (max 1.2 rad)
            
        if self._isBranches:
            # Control output
            command = int(target[24])-2
            out = np.zeros((4, 3), dtype=np.float32)  # modes x actions (controls)
            out[command,:] = target[:3]

            # Mask
            mask = np.zeros((4, 3), dtype=np.float32)
            mask[command,:] = 1

            # Speed input/output (max 90km/h)
            speed = np.array([target[10]/90,]).astype(np.float32)
            return img, command, speed, out.reshape(-1), mask.reshape(-1)

        else:
            # Control output
            command = int(target[24])-2

            # Speed input/output (max 90km/h)
            speed = np.array([target[10]/90,]).astype(np.float32)

            out = target[:3]
            return img, command, speed, out.reshape(-1)

    def __getitem__(self, idx):
        data_idx  = idx // FRAMES_PER_FILE
        file_idx  = idx  % FRAMES_PER_FILE
        file_name = self._files[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            # Image input
            img = np.array(h5_file['rgb'])[file_idx]
            img = self._transform(img)

            # Target dataframe
            target = np.array(h5_file['targets'])[file_idx]
            target = target.astype(np.float32)

            if self._isTrain:
                return self.  trainingRoutine(img,target)
            else:
                return self.evaluationRoutine(img,target)

