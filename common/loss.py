import numpy as np
import torch
import torch.nn as nn

class WeightedLoss(object):
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


class WeightedLossReg(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.lambda_action = setting.train.loss.lambda_action
        self.lambda_speed  = setting.train.loss.lambda_speed
        
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
            weight = weight.to(self.device)
            
            loss = loss.mul(weight)

        prediction['loss'] = loss
        return torch.mean(loss)


class MultitaskLoss(object):
    """ Constructor """
    def __init__(self,init,setting):
        # Parameters
        self.device = init.device
        self.lambda_desc   = setting.train.loss.lambda_desc
        self.lambda_action = setting.train.loss.lambda_action
        self.lambda_speed  = setting.train.loss.lambda_speed
        
        self.NLLLoss = nn.NLLLoss()

        # Weight
        self.weightLoss = np.array([setting.train.loss.lambda_steer, 
                                    setting.train.loss.lambda_gas  , 
                                    setting.train.loss.lambda_brake]).reshape(3,1)
        if setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device)
        self.decisionWeight = np.array( [0.,1.,2.] )
        self.decisionWeight = torch.from_numpy(self.decisionWeight).float().cuda(self.device) 

    def _getRawDecision(self,action):
        # decision: [cte,throttle,brake]
        decision = torch.where(action>0.05, 1., 0.)
        decision = decision.matmul(self.decisionWeight) # [0,1,2]

        # Check
        return torch.where(decision>2,0.,decision) 

    def eval(self,measure,prediction,weight=None):
        # Action loss
        action = torch.abs(measure['actions'] - prediction['actions'])
        action = action.matmul(self.weightLoss)

        # Cross entropy loss
        decision = self._getRawDecision(measure['actions'])
        action  += self.lambda_desc*self.NLLLoss(prediction['decision'],decision.long())

        # Speed regularization loss
        speed   = torch.abs(measure[ 'speed' ] - prediction[ 'speed' ])

        # Total loss
        loss = self.lambda_action*action + self.lambda_speed*speed

        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(-1,1)
            weight = weight.to(self.device)
            
            loss = loss.mul(weight)

        prediction['loss'] = loss
        return torch.mean(loss)

