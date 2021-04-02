import numpy as np

class SumTree(object):
    def __init__(self,n_nodes,val=None,limit=None):
        self.n_nodes = int( n_nodes )
        self.n_leaf  = int((n_nodes+1)/2 )
        self.limit   = (limit if limit is not None else self.n_leaf)

        # Samples Tree
        self._data = np.zeros( self.n_nodes )

        # Fill tree
        if val is not None:
            for i in range(self.limit):
                self.__setitem__(i,val)
                
    def _update(self,idx):
        son1 = self._data[2*idx + 1]
        son2 = self._data[2*idx + 2]
        self._data[idx] = son1 + son2
        # Root
        if idx == 0: return son1 + son2
        else: return self._update( int(np.ceil(idx/2)-1) ) 

    def __setitem__(self, idx, data):
        idx = idx + (self.n_leaf - 1) 
        self._data[idx] = data
        
        # Update priority
        n = int(np.ceil(idx/2)-1)
        self._update( n ) 
    def __getitem__(self, idx):
        idx = idx + (self.n_leaf - 1)
        return self._data[idx]
    
    def sum(self):
        return self._data[0]

    def search(self,value,node=0):
        # Root
        if node == 0 and value>=self._data[0]:
            return self.n_nodes - 1 # Last
        
        # Branches
        if node < self.n_leaf - 1:
            son1 = int(2*node + 1)
            son2 = int(2*node + 2)
            
            # Left
            if value < self._data[son1]:
                return self.search(   value  ,son1)

            # Right
            else:
                base = self._data[son1]
                return self.search(value-base,son2)
        else:
            return int(node)
            
