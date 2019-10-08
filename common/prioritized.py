import numpy as np

class PrioritizedSamples(object):
    def __init__(self,n,alpha=1.0,beta=0.9):
        self.n_samples = n
        self.n_leaf    = int(2**np.ceil(np.log2(n)))
        self.n_nodes   = 2*self.n_leaf - 1

        self.alpha = alpha
        self.beta  =  beta

        # Priority Tree 
        self.priorityPowAlpha = np.zeros( self.n_nodes )
        
    def _update(self,idx):
        son1 = self.priorityPowAlpha[2*idx + 1]
        son2 = self.priorityPowAlpha[2*idx + 2]
        self.priorityPowAlpha[idx] = son1 + son2
        # Root
        if idx == 0:
            return son1 + son2
        else:
            return self._update( int(np.ceil(idx/2)-1) )  

    def update(self,idx,p = None):
        idx = idx + (self.n_samples - 1)
        self.priorityPowAlpha[idx] = p**self.alpha
        
        return self._update( int(np.ceil(idx/2)-1) )

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
        
        # Index in data
        idx = idx - (self.n_samples - 1)
        return int(idx),weight
        
