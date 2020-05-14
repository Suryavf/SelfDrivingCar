import numpy as np
from common.prioritized import PrioritizedExperienceReplay 

class ReplayMemory(object):
    def __init__(self, n_buffer,len_state,len_action):
        # Parameters
        self.n_buffer      = n_buffer
        self.len_state     = len_state
        self.len_action    = len_action
        self.n_experiences = 0
        self.pointer       = 0

        # Buffer
        self.state     = np.zeros([n_buffer,len_state ],dtype=float)
        self.action    = np.zeros([n_buffer,len_action],dtype=float)
        self.reward    = np.zeros( n_buffer            ,dtype=float)
        self.new_state = np.zeros([n_buffer,len_state ],dtype=float)

        # Priority
        self.priority = PrioritizedExperienceReplay(n_buffer)

    def getBatch(self):
        idx,weight = self.priority.sample()
        return (self. state[idx],self.   action[idx],
                self.reward[idx],self.new_state[idx]), weight
        
    def size(self):
        return self.n_buffer
    def count(self):
        return self.n_experiences

    def add(self, state, action, reward, new_state, done):
        n = self.n_experiences 
        if  n < self.n_buffer:
            self.n_experiences += 1
        else:
            n = self.pointer
            self.pointer += 1        
            if self.pointer>=self.n_buffer:
                self.pointer = 0 
        self.state    [n] = state
        self.action   [n] = action
        self.reward   [n] = reward
        self.new_state[n] = new_state

    def erase(self):
        self.state     = np.zeros([self.n_buffer,self.len_state ],dtype=float)
        self.action    = np.zeros([self.n_buffer,self.len_action],dtype=float)
        self.reward    = np.zeros( self.n_buffer                 ,dtype=float)
        self.new_state = np.zeros([self.n_buffer,self.len_state ],dtype=float)
        self.n_experiences = 0
        self.pointer       = 0 
        
