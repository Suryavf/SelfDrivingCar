import numpy as np
import torch

class WeightedLossAct(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device

        # Weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        if setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device)

    def eval(self,measure,prediction,weight=None):
        loss = torch.abs(measure['actions'] - prediction['actions'])
        loss = loss.matmul(self.weightLoss)
        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(-1,1)
            weight = weight.to(self.device)
            
            loss = loss.mul(weight)
        
        prediction['loss'] = loss
        return torch.mean(loss)


class WeightedLossActSpeed(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.lambda_action = setting.train.loss.lambda_action
        self.lambda_speed  = setting.train.loss.lambda_speed
        self.sequence_len  = setting.train.sequence_len

        # Weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        if setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device)

    def eval(self,measure,prediction,weight=None):
        # Action loss
        action = torch.abs(measure['actions'] - prediction['actions'])
        action = action.matmul(self.weightLoss)

        # Speed loss
        speed   = torch.abs(measure[ 'speed' ] - prediction[ 'speed' ])
        
        # Total loss
        loss = self.lambda_action*action + self.lambda_speed*speed

        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(1,-1)
            ones   = np.ones([self.sequence_len,1])
            weight = ones.dot(weight).reshape([-1,1],order='F')
            weight = torch.from_numpy(weight).to(self.device)
            
            loss = loss.mul(weight)

        prediction['loss'] = loss
        return torch.mean(loss)


class PID(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.lambda_integral = 0.1
        self.lambda_derived  = 0.01
        
        # Weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        if setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device)


    def eval(self,measure,prediction,weight=None):
        Lprod = torch.abs(measure['actions'] - prediction['actions'])
        Ldriv = Lprod[1:,:] - Lprod[:-1,:]


        Lprod = Lprod.mean(0)

        #LossP = LossP.matmul(self.weightLoss)
        
