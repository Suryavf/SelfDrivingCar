import os
import glob
import h5py
import numpy as np
from   matplotlib import pyplot as plt
from   matplotlib.colors import LogNorm

# Figure list
use_corr = True # Beta correlation
use_dist = True # Beta distribution
use_fSim = True # Feature similarity

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

# Initialize beta correlation
if use_corr:
    Sx  = np.zeros([n_task,1,n_feature])
    Sxx = np.zeros([n_task,1,n_feature])
    Sxy = np.zeros([n_task,n_feature,n_feature])
    
# Initialize beta distribution
if use_dist:
    dist = np.zeros([n_task,n_feature,n_distr+1])
    
# Initialize feature similarity
if use_fSim:
    sumFtSim = np.zeros([n_task,n_feature,n_feature])

# Big loop
for n,f in enumerate(filesname):
    # Getting data
    print('Load file',f)
    with h5py.File(f, 'r') as h5_file:
        if use_corr or use_dist:
            beta = np.array(h5_file['beta'])
            beta = beta.squeeze()
            beta = beta.transpose(1,0,2)    # [task,batch,feature]
        if use_fSim:
            feature = np.array(h5_file['feature'])
            feature = feature.squeeze()     # [batch,task,depth,feature]

    # Beta correlation
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

    # Beta distribution
    if use_dist:
        # Compute histogram
        for t in range(n_task):
            for k in range(n_feature):
                hist,_ = np.histogram(beta[t,:,k],bins=n_distr,range=(0,1))
                dist[t,k] += hist
                
    # Feature similarity
    if use_fSim:
        # Compute direction
        mod = np.linalg.norm(feature, ord=2, axis=2, keepdims=True)
        direc = feature/mod     # [batch,task,depth,feature]
        
        # Compute similarity
        sumFtSim += np.matmul(direc.transpose(0,1,3,2),direc).mean(axis=0)
        ftSim = sumFtSim/(n+1)  # [task,n_feat,n_feat]


if use_dist:
    # Parameters
    umb      = int(n_feature/2)
    max_hist = dist.max()

    # Mode
    mode = dist.argmax(axis=2)*(1/n_distr)  # [task,feature]
    
    # Sort
    arg = list()
    for t in range(n_task):
        argZ = np.argsort(mode[t,:umb],kind='stable')
        argF = np.argsort(mode[t,umb:],kind='stable') + umb
        argZ = argZ[::-1]
        argF = argF[::-1]
        arg.append( np.concatenate([argZ,argF],axis=0) )
    arg = np.concatenate([arg])

    # Loop task
    for t in range(n_task):
        # Select data
        com = mode[t,arg[t]].reshape(n_feature,1)
        his = dist[t,arg[t]] + 1

        # Plot initialize
        fig = plt.figure(constrained_layout=False, figsize=(16,8))
        s = fig.add_gridspec(1, 2, width_ratios = [1,20])
        ax1 = fig.add_subplot(s[0,0])
        ax2 = fig.add_subplot(s[0,1])
        
        # Mode comparative
        ax1.matshow(com, cmap='coolwarm',vmin=0,vmax=com.max(),
                    extent=(0,1,0,n_feature))
        for i in range(n_feature): ax1.hlines(i,0,1, lw=0.5,ls=':')
        ax1.hlines(16,0,1)
        ax1.axis('off')
        
        # Histogram
        ax2.matshow(his, cmap='coolwarm', vmin=1, vmax=max_hist, norm=LogNorm(),
                    extent=(0,1000,0,n_feature))
        for i in range(n_feature): ax2.hlines(i,0,1000, lw=0.5,ls=':')
        ax2.hlines(umb,0,1000)
        ax2.set_aspect('auto')
        ax2.axis('off')
        
        # Save
        plt.savefig('betaHist_task%i.svg'%(t+1),quality=900)
else:
    # No testing
    arg = np.array(range(n_feature))
    arg = [arg for _ in range(n_task)]
    arg = np.concatenate([arg])    


if use_corr:
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(45,15)
    umb = int(n_feature/2)
    
    for t in range(n_task):
        axes[t].matshow(rxy[t,arg[t]][:,arg[t]], cmap='coolwarm',vmin=-1,vmax=1,
                        extent=(0,n_feature,0,n_feature))
        axes[t].axis('off')
        for i in range(n_feature): axes[t].hlines(i,0,n_feature, lw=0.5,ls=':')
        axes[t].hlines(umb,0,n_feature)
        for i in range(n_feature): axes[t].vlines(i,0,n_feature, lw=0.5,ls=':')
        axes[t].vlines(umb,0,n_feature)
    plt.savefig('betaCorr.svg',quality=900)
    
