import os
import glob

from   tqdm import tqdm
import pandas as pd
import numpy  as np
import carla

import torch
import torch.optim as optim
import torch.nn.functional as F
from   torch.utils.data        import DataLoader
from   torch.utils.tensorboard import SummaryWriter

import common.figures as F
import common.  utils as U
import common.carla_utils as cu
from   common.RAdam       import RAdam
from   common.Ranger      import Ranger
from   common.DiffGrad    import DiffGrad
from   common.DiffRGrad   import DiffRGrad
from   common.DeepMemory  import DeepMemory

import carla
from   benchmark import make_suite

# https://github.com/cjy1992/gym-carla
# https://github.com/carla-rl-gym/carla-rl
# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_utils.py
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# https://github.com/navneet-nmk/pytorch-rl
class Agent(object):
    """ Constructor """
    def __init__(self,init,setting):

        self.init    =    init
        self.setting = setting

        # Device
        self.device = self.init.device

        # Internal parameters
        self.state    = {}
        self.epoch    = 1
        self.codename = None

        # Paths
        self.           modelPath = None
        self.          figurePath = None
        self.figureSteerErrorPath = None
        self.  figureGasErrorPath = None
        self.figureBrakeErrorPath = None


        # Models
        if   self.setting.model ==         'DDPG': self.model = None
        elif self.setting.model ==         'D4PG': self.model = None
        elif self.setting.model == 'Experimental': self.model = None
        else:
            print("ERROR: mode no found (" + self.setting.model + ")")

        # Objects
        self.optimizer  = None
        self.reward     = None


    """ Check folders to save """
    def _checkFoldersToSave(self, name = None):
         # Root Path
        savedPath = self.setting.general.savedPath
        modelPath = os.path.join(savedPath,self.setting.model)
        if name is not None: self.codename = name
        else               : self.codename = U.nameDirectory()
        execPath  = os.path.join(modelPath,self.codename )
        U.checkdirectory(savedPath)
        U.checkdirectory(modelPath)
        U.checkdirectory( execPath)
        print("Execute %s model: %s\n"%(self.setting.model,self.codename))

        # Figures Path
        self.figurePath           = os.path.join(execPath,"Figure")
        self.figureSteerErrorPath = os.path.join(self.figurePath,"SteerError")
        self.figureGasErrorPath   = os.path.join(self.figurePath,  "GasError")
        self.figureBrakeErrorPath = os.path.join(self.figurePath,"BrakeError")

        U.checkdirectory(self.figurePath)
        U.checkdirectory(self.figureSteerErrorPath)
        U.checkdirectory(self.figureGasErrorPath  )
        U.checkdirectory(self.figureBrakeErrorPath)

        # Model path
        self.modelPath = os.path.join(execPath,"Model")
        U.checkdirectory(self.modelPath)


    """ Training state functions """
    def _state_reset(self):
        self.state = {}
    def _state_add(self,name,attr):
        self.state[name]=attr
    def _state_addMetrics(self,metr):
        self._state_add('steerMSE',metr[0])
        self._state_add(  'gasMSE',metr[1])
        self._state_add('brakeMSE',metr[2])
        if self.setting.boolean.speedRegression:
            self._state_add('speedMSE',metr[3])
    def _state_save(self,epoch):
        path = self.modelPath + "/model" + str(epoch) + ".pth"
        torch.save(self.state,path)


    """ Building """
    def build(self):
        self.model = self.model.float()
        self.model = self.model.to(self.device)

        # Optimizator
        if   self.setting.train.optimizer.type == "Adam"      : optFun = optim.Adam
        elif self.setting.train.optimizer.type == "RAdam"     : optFun = RAdam
        elif self.setting.train.optimizer.type == "Ranger"    : optFun = Ranger
        elif self.setting.train.optimizer.type == "DiffGrad"  : optFun = DiffGrad
        elif self.setting.train.optimizer.type == "DiffRGrad" : optFun = DiffRGrad
        elif self.setting.train.optimizer.type == "DeepMemory": optFun = DeepMemory
        else:
            txt = self.setting.train.optimizer.type
            raise NameError('ERROR 404: Optimizer no found ('+txt+')')
        self.optimizer = optFun(    self.model.parameters(),
                                    lr    =  self.setting.train.optimizer.learning_rate, 
                                    betas = (self.setting.train.optimizer.beta_1, 
                                             self.setting.train.optimizer.beta_2 ) )

    """ Validation metrics """
    def _metrics(self,measure,prediction):
        # Parameters
        max_steering = self.setting.preprocessing.max_steering

        # Measurements
        dev_Steer = measure['actions'][:,0] * max_steering
        dev_Gas   = measure['actions'][:,1]
        dev_Brake = measure['actions'][:,2]

        # Prediction
        # dev_SteerPred = prediction['actions'][:,0] * max_steering
        dev_GasPred   = prediction['actions'][:,1]
        dev_BrakePred = prediction['actions'][:,2]

        # Error
        dev_err = torch.abs(measure['actions'] - prediction['actions'])
        dev_SteerErr = dev_err[:,0] * max_steering
        dev_GasErr   = dev_err[:,1]
        dev_BrakeErr = dev_err[:,2]

        # Metrics
        metrics = dict()
        metrics['SteerError'] = dev_SteerErr.data.cpu().numpy()
        metrics[  'GasError'] = dev_GasErr  .data.cpu().numpy()
        metrics['BrakeError'] = dev_BrakeErr.data.cpu().numpy()

        metrics['Steer'] = dev_Steer.data.cpu().numpy()
        metrics[  'Gas'] = dev_Gas  .data.cpu().numpy()
        metrics['Brake'] = dev_Brake.data.cpu().numpy()

        # metrics['SteerPred'] = dev_SteerPred.data.cpu().numpy()
        metrics[  'GasPred'] = dev_GasPred  .data.cpu().numpy()
        metrics['BrakePred'] = dev_BrakePred.data.cpu().numpy()

        # Mean
        steerMean = np.mean(metrics['SteerError'])
        gasMean   = np.mean(metrics[  'GasError'])
        brakeMean = np.mean(metrics['BrakeError'])
        metricsMean = np.array([steerMean,gasMean,brakeMean])

        # Command control
        metrics['Command'] = measure['command'].data.cpu().numpy()

        return metrics,metricsMean

    """ Train """
    def _Train(self,env,epoch):
        # Parameters
        frames_per_episode = 4000 


        # Loop episode
        i = 0
        while i < frames_per_episode and not env.is_success() and not env.collided:
            env.tick()
            observation = env.get_observations()
            

            control = None
            env.apply_control(control)


    def execute(self):
        # Parameters
        n_episodes = self.setting.general.n_epoch

        with tqdm(total=n_episodes,leave=False) as pbar:
            for epoch in range(n_episodes):
                with make_suite('FullTown01-v1', port=2000, planner='new') as env:
                    # Setting envirtoment
                    start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
                    env_params = {  'weather'      : np.random.choice(list(cu.TRAIN_WEATHERS.keys())),
                                    'start'        : start,
                                    'target'       : target,
                                    'n_pedestrians': U.randint(0,20),
                                    'n_vehicles'   : U.randint(5,20),
                                }
                    env.init(**env_params)
                    env.success_dist = 5.0

                    # Train model
                    self._Train(env,epoch)
                
                    pbar. update()
                    pbar.refresh()
            pbar.close()
            
