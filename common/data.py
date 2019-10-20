import glob
import random
import numpy  as np
import pandas as pd
import torch
import h5py
from   torch.utils.data   import Dataset
from   torchvision        import transforms
from   imgaug             import augmenters as iaa
from   common.prioritized import PrioritizedSamples

from IPython.core.debugger import set_trace


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
    https://github.com/felipecode/imitation-learning-1
    https://github.com/carla-simulator/data-collector/blob/master/docs/dataset_format_description.md
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
class CoRL2017Dataset(object):
    def __init__(self, setting, files, train = True):
        # Boolean
        self.isTrain         = train
        self.isBranches      = setting.boolean.branches
        self.includeSpeed    = setting.boolean.multimodal or setting.boolean.speedRegression
        self.isTemporalModel = setting.boolean.temporalModel

        # Settings
        self.setting = setting
        self.framePerFile = self.setting.general.framePerFile

        # Files (paths)
        self.files = files

        # Objects
        self.transform = None
        
        # Data augmentation
        self.transformDataAug = transforms.RandomOrder([
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
                                    ])
        
        # Build data augmentation
        self.build()
        
        if self.isTemporalModel and self.isTrain:
            self.sequence_len   = setting.general.sequence_len
            self.slidingWindow  = 5 
            self.samplesPerFile = int( (self.framePerFile - self.sequence_len)/self.slidingWindow + 1 )
        else:
            self.samplesPerFile = self.framePerFile


    def build(self):
        trans = list()
        if   self.isTrain: trans.append(self.transformDataAug)
        trans.append(transforms.ToPILImage())
        if   self.setting.boolean.backbone ==   'CNN5': trans.append(transforms.Resize((92,196)))
        elif self.setting.boolean.backbone == 'ResNet': trans.append(transforms.Resize((96,192)))
        trans.append(transforms.ToTensor())
        self.transform = transforms.Compose(trans)


    def __len__(self):
        return self.samplesPerFile * len(self.files)

    def routine(self,img,target):
        # Parameters
        max_steering = self.setting.preprocessing.max_steering
        max_speed    = self.setting.preprocessing.max_speed
        inputs       = {}

        # Command control 
        command = int(target[24])-2
        if not self.isTrain:
            inputs['command'] = command

        # Frame
        inputs['frame'] = img

        # Actions
        target[0] = target[0]/max_steering   # Steering angle (max 1.2 rad)
        if self.isBranches: 
            actions = np.zeros((4, 3), dtype=np.float32)  # modes x actions (controls)
            actions[command,:] = target[:3]
        else:
            actions = target[:3]
        inputs['actions'] = actions.reshape(-1)

        # Mask
        if self.isBranches:
            mask = np.zeros((4, 3), dtype=np.float32)
            mask[command,:] = 1
            inputs[ 'mask' ] = mask.reshape(-1)
        
        # Speed input/output (max 90km/h)
        if self.includeSpeed or not self.isTrain:
            speed = np.array([target[10]/max_speed,]).astype(np.float32)
            inputs['speed'] = speed

        return inputs


    def __getitem__(self, idx):
        # File / Frame
        idx_file   = idx // self.framePerFile
        idx_sample = idx  % self.framePerFile
        file_name  = self.files[idx_file]
        
        # Read
        with h5py.File(file_name, 'r') as h5_file:
            # Image input
            img = np.array(h5_file['rgb'])[idx_sample]
            img = self.transform(img)

            # Target dataframe
            target = np.array(h5_file['targets'])[idx_sample]
            target = target.astype(np.float32)

            return self.routine(img,target) 
            

class GeneralDataset(Dataset):
    def __init__(self, dataset, IDs, weights = None):
        self.dataset = dataset
        self.    IDs = IDs
        self.weights = weights

    def __len__(self):
        return self.IDs.size
    
    def __getitem__(self,_idx):
        idx  = self.   IDs[_idx]
        data = self.dataset[idx]

        if self.weights is not None:
            return data,idx,self.weights[_idx]
        else:
            return data,idx
                        
