import numpy as np
import torch
import torch.nn as nn


""" Loss Function 
    --------------------------------------------------------------------
    * Input:
        - measure   : dict of real measurement
        - prediction: dict of prediction
        - weight    : bias for prioritized sampling
"""
class WeightedLoss(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.reg    = setting.train.loss.regularization

        # Weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        if self.reg:
            self.lambda_action = setting.train.loss.lambda_action
            self.lambda_speed  = setting.train.loss.lambda_speed
        if setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device)

    def eval(self,measure,prediction,weight=None):
        loss = torch.abs(measure['actions'] - prediction['actions'])
        loss = loss.matmul(self.weightLoss)

        # Speed regularization loss
        if self.reg:
            speed = torch.abs(measure[ 'speed' ] - prediction[ 'speed' ])
            loss = self.lambda_action*loss + self.lambda_speed*speed

        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(-1,1)
            weight = weight.to(self.device)
            
            loss = loss.mul(weight)
        
        prediction['loss'] = loss
        return torch.mean(loss)


class WeightedLossReg(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.reg    = setting.train.loss.regularization
        
        # Weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        if self.reg:
            self.lambda_action = setting.train.loss.lambda_action
            self.lambda_speed  = setting.train.loss.lambda_speed
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
            weight = weight.to(self.device)
            
            loss = loss.mul(weight)

        prediction['loss'] = loss
        return torch.mean(loss)


class MultitaskLoss(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.regularization = setting.train.loss.regularization
        self.use_decision   = False
        
        # Loss weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        
        # Regularization weight
        if self.regularization:
            self.lambda_action = setting.train.loss.lambda_action
            self.lambda_speed  = setting.train.loss.lambda_speed
        
        # Use brances
        if setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device)

        # Decision weight
        if self.use_decision:
            self.lambda_dec   = setting.train.loss.lambda_desc
            self.NLLLoss      = nn.NLLLoss()
            self.decisionWeight = np.array( [0.,1.,2.] )
            self.decisionWeight = torch.from_numpy(self.decisionWeight).float().cuda(self.device) 

    def _getRawDecision(self,action):
        # decision: [cte,throttle,brake]
        decision = torch.where(action>0.05, 1., 0.)
        decision = decision.matmul(self.decisionWeight) # [0,1,2]
        
        # Check
        return torch.where(decision>torch.tensor(2.).cuda(), torch.tensor(0.).cuda(), decision)

    def eval(self,measure,prediction,weight=None):
        # Action loss
        loss = torch.abs(measure['actions'] - prediction['actions'])
        loss = loss.matmul(self.weightLoss)

        # Cross entropy loss
        if self.use_decision:
            decision = self._getRawDecision(measure['actions'])
            loss    += self.lambda_dec*self.NLLLoss(prediction['manager'],decision.long())

        # Speed regularization loss
        if self.regularization:
            speed = torch.abs(measure[ 'speed' ] - prediction[ 'speed' ])
            loss  = self.lambda_action*loss + self.lambda_speed*speed

        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(-1,1)
            weight = weight.to(self.device)
            
            loss = loss.mul(weight)

        prediction['loss'] = loss
        return torch.mean(loss)
        
