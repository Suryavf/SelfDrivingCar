import os
import glob
import h5py
import numpy as np
from   matplotlib import pyplot as plt

# Figure list
use_corr = True
use_dist = True

# Parameters
path      = '/media/victor/Documentos/'
pathout   = '/media/victor/Documentos/'
n_feature =  32
n_distr   = 200
n_task    =   3
def getint(name):
    basename = name.partition('.')
    _, num = basename[0].split('resume')
    return int(num)
filesname = glob.glob(os.path.join(path,'*.sy'))
filesname.sort(key=getint)

# Initialize correlation
if use_corr:
    Sx  = np.zeros([n_task,1,n_feature])
    Sxx = np.zeros([n_task,1,n_feature])
    Sxy = np.zeros([n_task,n_feature,n_feature])

# Initialize distribution
if use_dist:
    dist = np.zeros(n_task,n_feature,n_distr+1)
    
# Big loop
for n,f in enumerate(filesname):
    # Getting data
    print('Load file',f)
    with h5py.File(f, 'r') as h5_file:
        beta = np.array(h5_file['beta'])
        beta = beta.squeeze()
        beta = beta.transpose(1,0,2)    # [task,batch,feature]

    # Correlation
    if use_corr:
        Sx  +=  beta    .mean(axis=1,keepdims=True)
        Sxx += (beta**2).mean(axis=1,keepdims=True)
        Sxy += np.matmul(beta.transpose(0,2,1),beta)/beta.shape[1]
        
        Sx_  = Sx /(n+1)
        Sxx_ = Sxx/(n+1)
        Sxy_ = Sxy/(n+1)
        
        σx   = np.sqrt( Sxx_ - Sx_**2 )
        SxSy = np.matmul(Sx_.transpose(0,2,1),Sx_)
        σxσy = np.matmul(σx .transpose(0,2,1),σx)
        cov  = Sxy_ - SxSy
        rxy  = cov/σxσy

    # Distribution
    if use_dist:
        # Compute histogram
        for t in range(n_task):
            for k in range(n_feature):
                dist[t,k] += np.histogram(beta[t,:,k],bins=n_distr,range=(0,1))

if use_corr:
    fig, axes = plt.subplots(nrows=1, ncols=n_task)
    fig.set_size_inches(15,5)
    for t in range(n_task):
        axes[t].matshow(rxy[t], cmap='coolwarm')
    plt.savefig(os.path.join(pathout,'betaCorr.svg'))
    

if use_dist:
    umb  = int(n_feature/2)
    mode = dist.argmax(axis=2)*(1/n_distr)  # [task,feature]
    
    # Sort
    argZ = mode[:,:umb].argsort(axis=1)
    argF = mode[:,umb:].argsort(axis=1) + umb
    arg  = np.concatenate([argZ,argF],axis=1)
    
    # Reorder
    dist = dist[arg] # [n_task,n_feature,n_distr+1]

    for t in range(n_task):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].matshow(mode[t].unsqueeze(1), cmap='coolwarm')
        axes[1].matshow(dist[t]             , cmap='coolwarm')
        plt.show()
        
