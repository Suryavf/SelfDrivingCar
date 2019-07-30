import sys
import contextlib

from os      import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
from   torch.utils.data import Dataset,DataLoader

import numpy as np
from   tqdm import tqdm

import common.figures as F

from common.data import CoRL2017Dataset as Dataset
from common.utils import iter2time
from common.utils import averager
from config import Config
from config import Global

# Solution DataLoader bug
# Ref: https://github.com/pytorch/pytorch/issues/973
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# =================
torch.multiprocessing.set_sharing_strategy('file_system')
# =================

# CUDA
device = torch.device('cuda:0')

# Settings
_global = Global()
_config = Config()

# Batch size
batch_size  = _config.batch_size

# Number of workers
num_workers = _global.num_workers

# Parameters
stepView  = _global.stepView

# Conditional (branches)
if _config.model in ['Codevilla18','Codevilla19']:
    __branches = True
else:
    __branches = False

# Multimodal (image + speed)
if _config.model in ['Multimodal','Codevilla18','Codevilla19']:
    __multimodal = True
else:
    __multimodal = False

# Speed regression
if _config.model in ['Codevilla19']:
    __speedReg = True
else:
    __speedReg = False

# Ref: https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write
class DummyFile(object):
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

def printLoss(i,loss,):
    print(i+1,":\tloss =",loss,"\t\t",iter2time(i))


""" Xavier initialization
    ---------------------
    Args:
        model: torch model
"""
def xavierInit(model):
    if isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0] # number of rows
        fan_in  = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        model.weight.data.normal_(0.0, variance)
"""
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight)
        m.bias.data.fill_(0.01)
"""


""" Weighted loss
    -------------
    Args:
        input : Model prediction
        target: Ground truth
"""
# Weight to weightedLoss()
if  _config.model in ['Basic', 'Multimodal', 'Codevilla18']:
    weightLoss = torch.Tensor([ _config.lambda_steer, 
                                _config.lambda_gas  , 
                                _config.lambda_brake]).float().cuda(device) 
elif _config.model in ['Codevilla19']:
    weightLoss = torch.Tensor([ _config.lambda_steer * _config.lambda_action, 
                                _config.lambda_gas   * _config.lambda_action, 
                                _config.lambda_brake * _config.lambda_action,
                                _config.lambda_speed ]).float().cuda(device) 
def weightedLoss(input, target):
    loss = torch.abs(input - target)
    loss = torch.mean(loss,0)

    return torch.sum(loss*weightLoss)


"""
Metrics
"""
def MSE(input, target):
    loss = (input - target) ** 2
    return torch.mean(loss,0)


""" Model prediction
    ----------------
    Predict target by input 
        * Input: model (nn.Module)
                 data  (tuple?)
        * Output: action: Ground truth
                  y_pred: prediction
"""
def runModel(model,data):
    action = None 
    y_pred = None

    # Codevilla18, Codevilla19
    if __multimodal and __branches:             
        frame, speed, action, mask = data

        mask   =   mask.to(device)
        frame  =  frame.to(device)
        speed  =  speed.to(device)
        action = action.to(device)

        y_pred = model(frame,speed,mask)
    
    # Multimodal
    elif __multimodal and not __branches:
        frame, speed, action = data

        frame  =  frame.to(device)
        speed  =  speed.to(device)
        action = action.to(device)

        y_pred = model(frame,speed)

    # Basic
    elif not __multimodal and not __branches:
        frame, action = data

        frame  =  frame.to(device)
        action = action.to(device)
        
        y_pred = model(frame)
    else:
        raise NameError('ERROR 404: Model no found')

    return action, y_pred


""" Train function
    --------------
    Global train function
        * Input: model     (nn.Module)
                 optimizer (torch.optim)
                 lossFunc  (function)
                 path      (path)
        * Output: total_loss (float) 
"""
def train(model,optimizer,lossFunc,path):
    
    # Acomulative loss
    running_loss = 0.0
    total_loss   = averager()

    # Data Loader
    loader = DataLoader( Dataset( path, train      =   True      , 
                                        branches   = __branches  ,
                                        multimodal = __multimodal,
                                        speedReg   = __speedReg ),
                        batch_size  = batch_size,
                        num_workers = num_workers)
    t = tqdm(iter(loader), leave=False, total=len(loader))
    # Train
    model.train()
    for i, data in enumerate(t,0):
        # Model execute
        action, output = runModel(model,data)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = lossFunc(output, action)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        runtime_loss = loss.item()
        running_loss += runtime_loss
        if i % stepView == (stepView-1):   # print every stepView mini-batches
            message = 'BatchTrain %i - loss=%.5f'
            t.set_description( message % ( i+1,running_loss/stepView ))
            t.refresh()
            running_loss = 0.0
        total_loss.update(runtime_loss)

    print("Epoch training loss:",total_loss.val())
    return total_loss.val()


"""
Validation function
"""
""" Validation function
    -------------------
    Global validation function
        * Input: model    (nn.Module)
                 lossFunc (function)
                 path     (path)
        * Output: total_loss (float) 
"""
def validation(model,lossFunc,epoch,path,figPath):
    # Acomulative loss
    running_loss = 0.0
    all_action  = list()
    all_speed   = list()
    all_command = list()
    lossValid = averager()
    
    max_speed    = _global.max_speed
    max_steering = _global.max_steering

    # Metrics
    if __speedReg:
        metrics = averager(4)   # Steer,Gas,Brake,Speed
    else:
        metrics = averager(3)   # Steer,Gas,Brake

    # Data Loader
    loader = DataLoader( Dataset( path, train      =   False     , 
                                        branches   = __branches  ,
                                        multimodal = __multimodal,
                                        speedReg   = __speedReg ),
                        batch_size  = batch_size,
                        num_workers = num_workers)
    n_loader = len(loader)
    pbar = tqdm(range(1,n_loader+1), leave=False, total=n_loader, desc = 'Validation' )
    
    # Model to evaluation
    model.eval()
    print("Execute validation")
    with torch.no_grad():
        i = 0
        for data in loader:
            
            if __branches:
                frame, command, speed, action, mask = data
            else:
                frame, command, speed, action = data

            # Codevilla18, Codevilla19
            if       __multimodal and     __branches: 
                data = frame, speed, action, mask
            elif     __multimodal and not __branches:
                data = frame, speed, action
            elif not __multimodal and not __branches:
                data = frame, action
            else:
                raise NameError('ERROR 404: Model no found')

            # Model execute
            action, output = runModel(model,data)
            
            # Calculate the loss
            loss  = lossFunc(output, action)
            runtime_loss = loss.item()
            running_loss += runtime_loss
            lossValid.update(runtime_loss)

            # Mean squared error
            err = MSE(output,action)
            err = err.data.cpu().numpy()
            metrics.update(err)
            
            # Save values
            all_action .append( output.data.cpu().numpy() )
            all_speed  .append(  speed  )
            all_command.append( command )

            if i % stepView == (stepView-1):   # print every stepView mini-batches
                message = 'BatchValid loss=%.5f'
                pbar.set_description( message % ( running_loss/stepView))
                pbar.refresh()
                running_loss = 0.0

            i += 1
            pbar.update()
    
    # Loss/metrics
    metrics      =    metrics.mean
    running_loss = lossValid.val()
    
    # Concatenate List
    all_command = np.concatenate(all_command,0)
    all_action  = np.concatenate(all_action ,0)
    all_speed   = np.concatenate(all_speed  ,0)

    # To real values
    metrics     [0] = metrics[0] * max_steering * max_steering
    all_action[:,0] = all_action[:,0] * max_steering
    all_speed       = all_speed       * max_speed
    if __speedReg:
        metrics [3] = metrics[3] * max_speed * max_speed

    # Print results
    print("Validation loss:",running_loss)
    if __speedReg:
        print("Steer:",metrics[0],"\tGas:",metrics[1],"\tBrake:",metrics[2],"\tSpeed:",metrics[3])
    else:
        print("Steer:",metrics[0],"\tGas:",metrics[1],"\tBrake:",metrics[2])
    
    # Save figures
    histogramPath = figPath + "/Histogram" + str(epoch) + ".png"
    scatterPath   = figPath + "/Scatter"   + str(epoch) + ".png"
    polarPath     = figPath + "/Polar"     + str(epoch) + ".png"
    F.saveScatterSteerSpeed     (all_action[:,0],all_speed,all_command, scatterPath )
    F.saveScatterPolarSteerSpeed(all_action[:,0],all_speed,all_command, polarPath )
    if __speedReg:
        F.saveHistogramSteerSpeed(all_action[:,0],all_speed,histogramPath)
    else:
        F.saveHistogramSteer     (all_action[:,0],          histogramPath)

    return running_loss,metrics
    
