import os
import glob
import h5py
import numpy as np
import cv2 as cv
from   matplotlib import pyplot as plt

# Figure list
use_corr = True

# Parameters
path      = '/media/victor/Documentos/'
n_feature = 32
def getint(name):
    basename = name.partition('.')
    _, num = basename[0].split('resume')
    return int(num)
filesname = glob.glob(os.path.join(path,'*.sy'))
filesname.sort()

Sx  = np.zeros(n_feature)
Sxx = np.zeros(n_feature)
Sxy = np.zeros(n_feature,n_feature)

# Big loop
for n,f in enumerate(filesname):
    # Getting data
    print('Load file',f)
    with h5py.File(f, 'r') as h5_file:
        beta = np.array(h5_file['beta'])

    # Correlation
    if use_corr:
        Sx  +=  beta    .mean(axis=0,keepdims=True)
        Sxx += (beta**2).mean(axis=0,keepdims=True)
        Sxy += np.dot(beta.T,beta)/len(beta)

        Sx_  = Sx /(n+1)
        Sxx_ = Sxx/(n+1)
        Sxy_ = Sxy/(n+1)
        
        σx   = np.sqrt( Sxx_ - Sx_**2 )
        SxSy = np.dot(Sx_.T,Sx_)
        σxσy = np.dot(σx.T,σx)
        cov  = Sxy_ - SxSy
        rxy  = cov/σxσy

        print('Correlation:')
        print(rxy)
        print('\n')
    
