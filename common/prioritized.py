import os
import h5py
import pickle
import numpy as np
from   common.graph import SumTree

class PrioritizedSamples(object):
    """ Constructor """
    def __init__(self,n_samples,alpha=1.0,beta=0.9,
                        betaLinear=False,betaPhase=50,
                        balance=False,c=1.0,
                        fill=False):
        # Parameters
        self.n_samples = n_samples
        self.alpha     =     alpha
        self.beta      =      beta

        self.n_leaf    = int(2**np.ceil(np.log2(n_samples)))
        self.n_nodes   = 2*self.n_leaf - 1

        if fill: _fill =  2.0
        else   : _fill = None

        # Samples Tree
        self.priority = SumTree( self.n_nodes,val=_fill,limit=n_samples )
        
        # Beta
        self.betaLinear = betaLinear
        if betaLinear: 
            self.beta_m  = (1.0-self.beta)/betaPhase
            self.beta_b  = self.beta
            self.n_iter  = 0

        # Upper Confidence Bound applied to trees (UCT)
        self.balance = balance
        if balance: 
            self.    exploParm = c
            self.sampleCounter = np.zeros( self.n_samples )
            self. totalCounter = 0
            self.UCT = SumTree( self.n_nodes,val=_fill,limit=n_samples )

    """ Save """
    def save(self,path='priority.ps'):
        # Save
        with h5py.File(path,"w") as f:
            # General
            dset = f.create_dataset( 'n_samples', data=self. n_samples)
            dset = f.create_dataset(     'alpha', data=self.     alpha)
            dset = f.create_dataset(      'beta', data=self.      beta)
            dset = f.create_dataset(    'n_leaf', data=self.    n_leaf)
            dset = f.create_dataset(   'n_nodes', data=self.   n_nodes)
            dset = f.create_dataset('betaLinear', data=self.betaLinear)
            dset = f.create_dataset(   'balance', data=self.   balance)

            # Priority tree
            dset = f.create_dataset('PS.n_nodes', data=self.priority.n_nodes)
            dset = f.create_dataset('PS.n_leaf' , data=self.priority. n_leaf)
            dset = f.create_dataset('PS.limit'  , data=self.priority.  limit)
            dset = f.create_dataset('PS.data'   , data=self.priority.  _data)

            if self.betaLinear: 
                dset = f.create_dataset('beta_m', data=self.beta_m)
                dset = f.create_dataset('beta_b', data=self.beta_b)
                dset = f.create_dataset('n_iter', data=self.n_iter)

            if self.balance: 
                dset = f.create_dataset('exploParm'    , data=self.    exploParm)
                dset = f.create_dataset('sampleCounter', data=self.sampleCounter)
                dset = f.create_dataset('totalCounter' , data=self. totalCounter)
                
                # UCT
                dset = f.create_dataset('UCT.n_nodes', data=self.UCT.n_nodes)
                dset = f.create_dataset('UCT.n_leaf' , data=self.UCT. n_leaf)
                dset = f.create_dataset('UCT.limit'  , data=self.UCT.  limit)
                dset = f.create_dataset('UCT.data'   , data=self.UCT.  _data)

    """ Load """
    def load(self,path='priority.ps'):
        _, extn = os.path.splitext(path)
        
        # Pickle file
        if extn == ".pck":
            with open(path, 'rb') as handle:
                p = pickle.load(handle)

            self.priority = p['priority']
            if self.balance: 
                self.sampleCounter = p['sampleCounter']
                self.totalCounter  = p[ 'totalCounter']
                self.UCT           = p          ['UCT']

        # H5py
        else:
            with h5py.File(path, 'r') as h5file:
                self.n_samples  = h5file[ 'n_samples'][()]
                self.alpha      = h5file[     'alpha'][()]
                self.beta       = h5file[      'beta'][()]
                self.n_leaf     = h5file[    'n_leaf'][()]
                self.n_nodes    = h5file[   'n_nodes'][()]
                self.betaLinear = h5file['betaLinear'][()]
                self.balance    = h5file[   'balance'][()]

                # Priority tree
                self.priority.n_nodes = h5file['PS.n_nodes'][()]
                self.priority. n_leaf = h5file['PS.n_leaf' ][()]
                self.priority.  limit = h5file['PS.limit'  ][()]
                self.priority.  _data = np.array(h5file['PS.data'])
                
                if self.betaLinear: 
                    self.beta_m = h5file['beta_m'][()]
                    self.beta_b = h5file['beta_b'][()]
                    self.n_iter = h5file['n_iter'][()]
                    
                if self.balance: 
                    self.exploParm     = h5file[   'exploParm'][()]
                    self.totalCounter  = h5file['totalCounter'][()]
                    self.sampleCounter = np.array(h5file['sampleCounter'])
                    
                    # UCT
                    self.UCT.n_nodes = h5file['UCT.n_nodes'][()]
                    self.UCT.n_leaf  = h5file['UCT.n_leaf' ][()]
                    self.UCT.limit   = h5file['UCT.limit'  ][()]
                    self.UCT._data   = np.array(h5file['UCT.data'])
                    
        
    """ Step """
    def step(self):
        if self.betaLinear: 
            self.beta = self.beta_b + self.beta_m*self.n_iter
            self.beta = min(self.beta,1.0)
            self.n_iter += 1

        # Update UCT
        if self.balance:
            for idx in range(self.n_samples):
                p = self.priority[idx]/self.priority.sum()
                u = self.exploParm*np.sqrt( np.log(self.totalCounter) / ( 1 + self.sampleCounter[idx]) )
                self.UCT[idx] = p+u
                

    """ Functions """
    def update(self,idx,p = None):
        # Prioritized sampling
        self.priority[idx] = p**self.alpha

        # UCT
        if self.balance:
            self.sampleCounter[idx]+=1
            self. totalCounter     +=1

            p = self.priority[idx]/self.priority.sum()
            u = self.exploParm*np.sqrt( np.log(self.totalCounter) / ( 1 + self.sampleCounter[idx]) )

            self.UCT[idx] = p+u

    
    """ Get sample """
    def sample(self):
        if self.balance: tree = self.UCT
        else           : tree = self.priority

        # Roulette
        sp = np.random.uniform()
        sp = sp * tree.sum()

        # Index in pow(priority,alpha)
        idx = tree.search(sp)
        idx = idx - (self.n_leaf - 1)

        if self.beta > 0:
            # Probability
            prob = tree[idx]/tree.sum()
            # Importance-sampling (IS) weights
            weight = ( 1/(self.n_samples*prob) )**self.beta
        else:
            weight = 1 
        
        return int(idx),weight
        

class PrioritizedExperienceReplay(object):
    """ Constructor """
    def __init__(self,n_samples,alpha=1.0,beta=0.9,
                        betaLineal=True,betaPhase=50,
                        UCB=False,c=0.0):
        # Parameters
        self.n_samples = n_samples
        self.alpha     =     alpha
        self.beta      =      beta
        self.e         =      1e-9

        self.n_leaf    = int(2**np.ceil(np.log2(n_samples)))
        self.n_nodes   = 2*self.n_leaf - 1

        # Samples Tree
        self.priorityPowAlpha = np.zeros( self.n_nodes ,dtype=float)
        self.max_weight       = 1.0

        # Beta
        self.betaLineal = betaLineal
        if not betaLineal: 
            self.beta_m  = (1.0-self.beta)/betaPhase
            self.beta_b  = self.beta
            self.n_iter  = 0

        # Upper Confidence Bound applied to trees (UCT)
        self.UCB = UCB
        if UCB: 
            # Samples count
            self.c = c
            self.sampleFame    = np.zeros( self.n_nodes ,dtype=float)  
            self.sampleFame[0] = 1.0 

            # Full UCT
            self.UCT = np.zeros( self.n_nodes ,dtype=float)  
            self.UCT[0] = 1.0

    """ Save """
    def save(self,path='priority.pck'):
        p = dict()
        p['priority'] = self.priorityPowAlpha
        if self.UCB: 
            p['sample'] = self.sampleFame
        # Save
        with open(path, 'wb') as handle:
            pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """ Load """
    def load(self,path='priority.pck'):
        with open(path, 'rb') as handle:
            p = pickle.load(handle)
        self.priorityPowAlpha = p['priority']
        self.sampleFame       = p[  'sample']
        
    """ Update """
    def _update(self,idx):
        son1 = self.priorityPowAlpha[2*idx + 1]
        son2 = self.priorityPowAlpha[2*idx + 2]
        self.priorityPowAlpha[idx] = son1 + son2
        # Root
        if idx == 0: return son1 + son2
        else: return self._update( int(np.ceil(idx/2)-1) ) 
    def _like(self,idx):
        son1 = self.sampleFame[2*idx + 1]
        son2 = self.sampleFame[2*idx + 2]
        self.sampleFame[idx] = son1 + son2
        # Root
        if idx == 0: return son1 + son2
        else: return self._like( int(np.ceil(idx/2)-1) ) 
    def update(self,idx,p = None):
        idx = idx + (self.n_leaf - 1)
        n = int(np.ceil(idx/2)-1)

        # Priority
        p = abs(p) + self.e

        # Probability
        self.priorityPowAlpha[idx] = p**self.alpha
        value = self._update( n )

        # Sample count
        if self.UCB: 
            self.sampleFame  [idx] += 1
            self._like( n )

        # Compute probability
        priorSamp = self.priorityPowAlpha[idx]/self.priorityPowAlpha[0]
        if self.UCB:
            _uct  = self.c*np.sqrt( np.log(self.sampleFame[0]) / ( 1 + self.sampleFame[idx]) )
            self.UCT[idx] = priorSamp + _uct

        return value

    """ Get sample """
    def _search(self,value,node):
        # Root
        if node == 0 and value>=self.priorityPowAlpha[0]:
            return self.n_nodes - 1 # Last
        
        # Branches
        if node < self.n_samples - 1:
            son1 = int(2*node + 1)
            son2 = int(2*node + 2)
            
            # Left
            if value < self.priorityPowAlpha[son1]:
                return self._search(   value  ,son1)

            # Right
            else:
                base = self.priorityPowAlpha[son1]
                return self._search(value-base,son2)
        else:
            return node
    def sample(self):
        # Roulette
        s = np.random.uniform()
        s = int(s * self.priorityPowAlpha[0])

        # Index in pow(priority,alpha)
        idx = self._search(s,0)
        if self.beta > 0:
            # Probability
            p = self.priorityPowAlpha[idx]/self.priorityPowAlpha[0]
            # Importance-sampling (IS) weights
            weight = ( 1/(self.n_samples*p) )**self.beta
        else:
            weight = 1
        
        # Update beta
        if  self.betaLineal: 
            self.beta = self.beta_b + self.beta_m*self.n_iter
            self.beta = min(self.beta,1.0)
            self.n_iter += 1

        # Index in data
        idx = idx - (self.n_leaf - 1)
        return int(idx),weight
        