import glob
import random
import numpy as np
import torch
import h5py
from   torch.utils.data   import Dataset
from   torchvision        import transforms
from   imgaug             import augmenters as iaa
from   common.prioritized import PrioritizedSamples


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
class CoRL2017Dataset(Dataset):
    def __init__(self, setting, train = True, exploration = False):
        # Boolean
        self._isTrain         = train
        self._isBranches      = setting.boolean.branches
        self._includeSpeed    = setting.boolean.multimodal or setting.boolean.speedRegression
        self._isTemporalModel = setting.boolean.temporalModel

        # Settings
        self.setting = setting
        self.framePerFile = self.setting.general.framePerFile

        # Files (paths)
        self.  trainingFiles = glob.glob(setting.general.trainPath+'*.h5')
        self.validationFiles = glob.glob(setting.general.validPath+'*.h5')
        self.  trainingFiles.sort()
        self.validationFiles.sort()

        if train: self.files = self.  trainingFiles
        else    : self.files = self.validationFiles

        # Objects
        self._transform = None
        
        # Data augmentation
        self._transformDataAug = transforms.RandomOrder([
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

        # Last sample (update prorities)
        self.lastSample = 0

        # Temporal models
        self._id_sample    = 0
        self._tempCount    = 0
        self._stepWindow   = 5
        self._sequence_len = setting.train.sequence_len 
        self._pointer      = setting.train.sequence_len 
        if self._isTemporalModel:
            n = (self.framePerFile - self._sequence_len + 1) * len(self.files)
        else:
            n = self.framePerFile * len(self.files)

        # Priorized
        self. prioritized = PrioritizedSamples( n )
        self._exploration = exploration

    def train(self):
        self.files = self.  trainingFiles
        self._isTrain = True
        self.build()
    def eval(self):
        self.files = self.validationFiles
        self._isTrain = False
        self.build()

    def build(self):
        if self._isTrain:
            self._transform = transforms.Compose([  self._transformDataAug,
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((92,196)),#(96,192)
                                                    transforms.ToTensor()])
        else:
            self._transform = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((92,196)),#(96,192)
                                                  transforms.ToTensor(),])

    def exploration(self,opt):
        self._exploration = opt

    def update(self,priority):
        self.prioritized.update(self.lastSample,priority)

    def __len__(self):
        if self._isTemporalModel:
            samplesPerFile = int( (self.framePerFile - self._sequence_len)/self._stepWindow + 1 )
        else:
            samplesPerFile = self.framePerFile
        return samplesPerFile * len(self.files)

    def trainingRoutine(self,img,target):
        max_steering = self.setting.preprocessing.max_steering
        max_speed    = self.setting.preprocessing.max_speed

        # Steering angle
        target[0] = target[0]/max_steering   # Angle (max 1.2 rad)
            
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
                speed = np.array([target[10]/max_speed,]).astype(np.float32)
                return img, speed, out.reshape(-1), mask.reshape(-1)
            else:
                return img, out.reshape(-1), mask.reshape(-1)

        else:
            out = target[:3]

            if self._includeSpeed:
                # Speed input/output (max 90km/h)
                speed = np.array([target[10]/max_speed,]).astype(np.float32)
                return img, speed, out.reshape(-1)
            else:
                return img, out.reshape(-1)

        
    def evaluationRoutine(self,img,target):
        max_steering = self.setting.preprocessing.max_steering
        max_speed    = self.setting.preprocessing.max_speed

        # Steering angle
        target[0] = target[0]/max_steering   # Angle (max 1.2 rad)
        
        if self._isBranches:
            # Control output
            command = int(np.array(target[24]-2))
            out = np.zeros((4, 3), dtype=np.float32)  # modes x actions (controls)
            out[command,:] = target[:3]

            # Mask
            mask = np.zeros((4, 3), dtype=np.float32)
            mask[command,:] = 1

            # Speed input/output (max 90km/h)
            speed = np.array([target[10]/max_speed,]).astype(np.float32)
            return img, command, speed, out.reshape(-1), mask.reshape(-1)
        else:
            # Control output
            command = np.array(target[24]-2)

            # Speed input/output (max 90km/h)
            speed = np.array([target[10]/max_speed,]).astype(np.float32)

            out = target[:3]
            return img, command, speed, out.reshape(-1)


    """ Data position for sampling
        --------------------------
        Priority memory:
            - No temporal data: n_samples = n_files * framePerFile
                                buffer: n_samples x [      1      frame]
            - Temporal data:    n_samples = n_files *(framePerFile - sequence_len + 1)
                                buffer: n_samples x [sequence_len frame]

    """
    def dataPosition(self,idx):
        # Samples per file
        if self._isTemporalModel:
            samplesPerFile = int( (self.framePerFile - self._sequence_len)/self._stepWindow + 1 )
        else:
            samplesPerFile = self.framePerFile

        # Priorized mode
        if not self._exploration:
            idx,weight = self.prioritized.sample()

        # Save last sample
        self.lastSample = idx

        idx_file   = idx // samplesPerFile
        idx_sample = idx  % samplesPerFile
        
        file_name = self.files[idx_file]
        idx_frame = idx_sample * self._stepWindow

        return file_name, idx_frame

    def getOneExample(self,file_name,file_idx):
        # Read
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


    def __getitem__(self, idx):
        # Data position
        file_name, idx = self.dataPosition(idx)

        if self._isTemporalModel and self._isTrain:
            sample = None
            # Get sample
            for i in range(self._sequence_len):
                # Get one example
                s = self.getOneExample(file_name,idx+i)
                # To sample (temporal)
                if i == 0: sample = [list() for _ in range(len(s))]
                for j, d in enumerate(s): sample[j].append( d )
            # Stack list
            for i, d in enumerate(sample):
                sample[i] = np.stack(d)

            return tuple(sample)
        else:
            return self.getOneExample(file_name,idx)
            
