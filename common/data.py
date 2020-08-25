import os
import random
import numpy  as np
import pandas as pd
import h5py
from   torch.utils.data   import Dataset
from   torchvision        import transforms
from   imgaug             import augmenters as iaa

from IPython.core.debugger import set_trace


class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)

""" CoRL2017 Dataset
    ----------------
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
        self.framePerFile  = self.setting.general. framePerFile
        self.slidingWindow = self.setting.general.slidingWindow

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
                                    RandomTransWrapper( seq=iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.5),
                                                        p=0.09),
                                    ])
        
        # Build data augmentation
        self.build()
        
        if self.isTemporalModel and self.isTrain:
            self.sequence_len   = setting.general.sequence_len
            self.slidingWindow  = setting.general.slidingWindow
            self.samplesPerFile = int( (self.framePerFile - self.sequence_len)/self.slidingWindow + 1 )
        else:
            self.samplesPerFile = self.framePerFile


    def build(self):
        trans = list()
        if   self.isTrain: trans.append(self.transformDataAug)
        trans.append(transforms.ToPILImage())
        trans.append(transforms.Resize(self.setting.boolean.shape))
        trans.append(transforms.ToTensor())
        self.transform = transforms.Compose(trans)


    def generateIDs(self,isTrainsequence=True):
        # Temporal
        # IDs = [file 1][file 2]....
        # len([file 1]) = sequence_len* int( (framePerFile-sequence_len)/slidingWindow + 1 ) 
        if isTrainsequence:
            framePerFile  = self.framePerFile
            sequence_len  = self.sequence_len
            slidingWindow = self.slidingWindow
            n_files       = len(self.files)
            n_samples     = n_files*int( (framePerFile-sequence_len)/slidingWindow + 1 )

            IDs = slidingWindow*np.array( range(n_samples) )
            IDs = [ np.array(range(idx,idx+sequence_len)) for idx in IDs if (idx%framePerFile) < (framePerFile-sequence_len)+1]
            IDs = np.concatenate(IDs)

        # No temporal
        else:
            n_samples = len(self.files)*self.framePerFile
            IDs = np.array( range(n_samples) )

        return IDs.astype(int)

    """ Sample to sampleFile-ID vector """
    def sample2Idx(self,arr):
        framePerFile  = self.framePerFile
        sequence_len  = self.sequence_len
        slidingWindow = self.slidingWindow
        k1 = sequence_len* int( (framePerFile-sequence_len)/slidingWindow + 1 )
        k2 = sequence_len-slidingWindow
        arr = np.array([ slidingWindow*x + int(x/k1)*k2 for x in arr ])
        return arr.astype(int)

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


    def __getitem__(self, idx, filename=None):
        # File / Frame
        if filename is None:
            idx_file   = idx // self.framePerFile
            idx_sample = idx  % self.framePerFile 
            filename   = self.files[idx_file]
        else:
            idx_file = idx

        # Read
        with h5py.File(filename, 'r') as h5_file:
            # Image input [88,200,3]
            img = np.array(h5_file['rgb'])[idx_sample]
            img = self.transform(img)

            # Target dataframe
            target = np.array(h5_file['targets'])[idx_sample]
            target = target.astype(np.float32)

            return self.routine(img,target) 


""" File Tree (CARLA 100)
    ----------------
    Tree for [prioritized] samples. Each leaf correspond to a training file.
    IMPORTANTE: El codigo asume sequence_len = slidingWindow
"""
class FileTree(object):
    """ Constructor """
    def __init__(self,path,setting):
        # Read data
        data = pd.read_csv(os.path.join(path,'index.csv'))
        n_files = len(data)

        self.n_files = n_files
        self.n_leaf  = int(2**np.ceil(np.log2(n_files)))
        self.n_nodes = 2*self.n_leaf - 1
        
        if setting.boolean.temporalModel:
            self.sequence_len   = setting.general.sequence_len
            self.slidingWindow  = setting.general.slidingWindow

        # Samples Tree
        self._tree = np.zeros( self.n_nodes )

        # Initialize
        self.file = data['file'].to_list()
        for idx,value in enumerate(data['n'].to_list()):
            value = int(np.floor(value/self.sequence_len)*self.sequence_len)
            self.update(idx,value)

    def __len__(self):
        return int(self._tree[0])

    """ Update """
    def _update(self,idx):
        son1 = self._tree[2*idx + 1]
        son2 = self._tree[2*idx + 2]
        self._tree[idx] = son1 + son2
        # Root
        if idx == 0: return son1 + son2
        else: return self._update( int(np.ceil(idx/2)-1) ) 
    def update(self,idx,value):
        idx = idx + (self.n_leaf - 1)
        self._tree[idx] = value
        
        # Update fame
        n = int(np.ceil(idx/2)-1)
        return self._update( n )

    """ Previous account """
    def _father(self,node):
        # Is it right branch?
        _right = (node%2 == 0)
        # Where is dad?
        _dad = int( (node-1)/2 )
        return _dad, _right
    def _previousSum(self,node,sum=0):
        # Root
        if node == 0: return sum
        # Calcule father
        father, right = self._father(node)
        # Accumulate sum 
        if right: sum += self._tree[2*father + 1]
        return self._previousSum(father,sum)
    
    """ Get sample """
    def _search(self,value,node=0):
        # Root
        if node == 0 and value>=self._tree[0]:
            return self.n_nodes - 1 # Last
        
        # Branches
        if node < self.n_files - 1:
            son1 = int(2*node + 1)
            son2 = int(2*node + 2)
            
            # Left
            if value < self._tree[son1]:
                return self._search(   value  ,son1)
            # Right
            else:
                base = self._tree[son1]
                return self._search(value-base,son2)
        else:
            return node

    def sample(self, idsample=None):
        
        # Roulette
        if idsample is None:
            idsample = np.random.uniform()
            idsample = idsample * self._tree[0]
            # idsample = int(idsample/self.sequence_len)*self.sequence_len
        # idsample to [general position]
        # else:
            # idsample = idsample * self.sequence_len # Position of frame
            
        # Find idsample [general position]
        idx  = self._search(idsample)
        prev = self._previousSum(idx)
        pos  = idsample - prev
        
        # File name
        idx = idx - (self.n_leaf - 1)  # Index in data
        filename = self.file[idx]
        
        # Return file and frame position in file
        # Only one frame!!! (no sequence)
        return int(pos),filename


""" CARLA 100 Dataset
    ----------------
    Data generator for Carla Dataset.
    Ref:

"""
class CARLA100Dataset(object):
    def __init__(self, setting, path, train = True):
        # Boolean
        self.isTrain         = train
        self.isBranches      = setting.boolean.branches
        self.includeSpeed    = setting.boolean.multimodal or setting.boolean.speedRegression
        self.isTemporalModel = setting.boolean.temporalModel

        # Settings
        self.setting = setting
        self.path    = path
        self.framePerFile = self.setting.general.framePerFile

        # Files (paths)
        self.files = FileTree(path,setting) # = fileindex

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
                                    RandomTransWrapper( seq=iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.5),
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
        trans.append(transforms.Resize(self.setting.boolean.shape))
        trans.append(transforms.ToTensor())
        self.transform = transforms.Compose(trans)

    def generateIDs(self):
        IDs = np.array( range( len(self.files) ) )
        return IDs.astype(int)

    """ Sample to sampleFile-ID vector """
    def sample2Idx(self,IDs):
        IDs = self.slidingWindow*IDs
        return IDs.astype(int)

    def __len__(self):
        return len( self.files )

    def routine(self,img,_actions,command,speed):
        # Parameters
        max_steering = self.setting.preprocessing.max_steering
        max_speed    = self.setting.preprocessing.max_speed
        inputs       = {}

        # Command control 
        command = int(command)-2
        if not self.isTrain:
            inputs['command'] = command

        # Frame
        inputs['frame'] = img

        # Actions
        _actions[0] = _actions[0]/max_steering   # Steering angle (max 1.2 rad)
        if self.isBranches: 
            actions = np.zeros((4, 3), dtype=np.float32)  # modes x actions (controls)
            actions[command,:] = _actions
        else:
            actions = _actions
        inputs['actions'] = actions.reshape(-1)

        # Mask
        if self.isBranches:
            mask = np.zeros((4, 3), dtype=np.float32)
            mask[command,:] = 1
            inputs[ 'mask' ] = mask.reshape(-1)
        
        # Speed input/output (max 90km/h)
        if self.includeSpeed or not self.isTrain:
            speed = np.array([speed/max_speed,]).astype(np.float32)
            inputs['speed'] = speed

        return inputs


    def __getitem__(self, idx):
        # File / Frame
        idx_sample,filename = self.files.sample(idx)
        
        # Read
        with h5py.File(os.path.join(self.path,filename), 'r') as h5_file:
            # Image input [88,200,3]
            img = np.array(h5_file['rgb'])[idx_sample]
            img = self.transform(img)

            # Target dataframe
            actions = np.array(h5_file['actions'])[idx_sample]
            actions = actions.astype(np.float32)

            # Command
            command = np.array(h5_file['command'])[idx_sample]
            command = command.astype(np.float32)

            # Velocity
            speed = np.array(h5_file['velocity'])[idx_sample]
            speed = speed.astype(np.float32)

            return self.routine(img,actions,command,speed) 
            

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
                        