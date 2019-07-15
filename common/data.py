import numpy as np
import torch
import h5py
from   torch.utils.data import Dataset

class CoRL2017Dataset(Dataset):
    """ Constructor """
    def __init__(self,path,complete=False):
        # Read
        self._d = h5py.File(path, 'r')
        self._frame    = self._d[ "frame"]
        self._output   = self._d["output"]
        self._len      = self._getLength(path) 
        self._complete = complete
        
        if complete:
            self._speed   = self._d[  "speed"]
            #self._command = self._d["command"]
    
    """ Get length by name """
    def _getLength(self,file):
        for i in range(len(file)-1,-1,-1):
            if file[i] == "_":
                start = i + 1
                break
        
        for i in range(len(file)-1,-1,-1):
            if file[i] == ".":
                end = i
                break
        
        return int(file[start:end])
    
    """ Length """
    def __len__(self):
        return self._len
    
    """ Delete object """
    def __del__(self):
        self._d.close()

    """ Get item """
    def __getitem__(self,idx):
        
        frame = self._frame[idx]
        frame = np.moveaxis(frame,2,0)
        frame = torch.from_numpy(frame).float()/255.0
        
        if self._complete:
            speed = self._speed  [idx]
            speed = torch.from_numpy(speed).float()
            
            #command = self._command[idx]
            #command = torch.from_numpy(command).float()

            output = self._output[idx,0:3]
            output = torch.from_numpy(output).float()

            #return frame,speed,command,output
            return frame,speed,output

        else:
            output = self._output[idx,0:3]
            output = torch.from_numpy(output).float()

            return frame,output