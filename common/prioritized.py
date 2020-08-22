import pickle
import  numpy as np

class PrioritizedSamples(object):
    """ Constructor """
    def __init__(self,n_samples,alpha=1.0,beta=0.9,
                        betaUniform=True,betaPhase=50,
                        UCB=False,c=1.0,
                        fill=False):
        # Parameters
        self.n_samples = n_samples
        self.alpha     =     alpha
        self.beta      =      beta

        self.n_leaf    = int(2**np.ceil(np.log2(n_samples)))
        self.n_nodes   = 2*self.n_leaf - 1

        # Samples Tree
        self.priorityPowAlpha = np.zeros( self.n_nodes )
        
        # Fill tree
        if fill:
            for i in range(self.n_leaf):
                idx = i + (self.n_leaf - 1)
                self.priorityPowAlpha[idx] = 2.0
                
                n = int(np.ceil(idx/2)-1)
                self._update( n )

        # Beta
        self.betaUniform = betaUniform
        if not betaUniform: 
            self.beta_m  = (1.0-self.beta)/betaPhase
            self.beta_b  = self.beta
            self.n_iter  = 0

        # Upper Confidence Bound (UCB)
        self.UCB = UCB
        if UCB: 
            self.c = c
            self.sampleFame    = np.zeros( self.n_nodes )  
            self.sampleFame[0] = 1.0 

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

    """ Step """
    def step(self):
        if not self.betaUniform: 
            self.beta = self.beta_b + self.beta_m*self.n_iter
            self.beta = min(self.beta,1.0)
            self.n_iter += 1

    """ Functions """
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

        if self.UCB: value = p**self.alpha + self.c*np.sqrt( np.log(self.sampleFame[0]) / ( 1 + self.sampleFame[idx]) )
        else       : value = p**self.alpha 
        
        self.priorityPowAlpha[idx] = value
        
        if self.UCB:
            self.sampleFame[idx] += 1
        
        # Update fame
        n = int(np.ceil(idx/2)-1)
        if self.UCB:
            self._like( n )

        # Update priority
        return self._update( n )
    
    """ Get sample """
    def _search(self,value,node):
        # Root
        if node == 0 and value>=self.priorityPowAlpha[0]:
            return self.n_nodes - 1 # Last
        
        # Branches
        if node < self.n_leaf - 1:
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
        
        # Index in data
        idx = idx - (self.n_leaf - 1)
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

        # Upper Confidence Bound (UCB)
        self.UCB = UCB
        if UCB: 
            self.c = c
            self.sampleFame    = np.zeros( self.n_nodes ,dtype=float)  
            self.sampleFame[0] = 1.0 

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
        idx = idx + (self.n_samples - 1)

        # Priority
        p = abs(p) + self.e

        # Probability
        if self.UCB: value = p**self.alpha + self.c*np.sqrt( np.log(self.sampleFame[0]) / ( 1 + self.sampleFame[idx]) )
        else       : value = p**self.alpha 
        
        # Update buffer
        self.priorityPowAlpha[idx] = value
        if self.UCB: 
            self.sampleFame[idx] += 1
        
        # Update fame
        n = int(np.ceil(idx/2)-1)
        if self.UCB: self._like( n )

        # Update priority
        return self._update( n )

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
        