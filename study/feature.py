import os
import glob
import h5py
import numpy as np
from   matplotlib import pyplot as plt

# Figure list
use_sim = True

# Parameters
path      = '/media/victor/Documentos/'
pathout   = '/media/victor/Documentos/'
n_feature = 32
n_distr   = 200
n_task    =  3
def getint(name):
    basename = name.partition('.')
    _, num = basename[0].split('resume')
    return int(num)
filesname = glob.glob(os.path.join(path,'*.sy'))
filesname.sort(key=getint)

# Initialize similarity study (direction)
if use_sim:
    sim = np.zeros(n_task,n_feature,n_feature)

# Big loop
for n,f in enumerate(filesname):
    # Getting data
    print('Load file',f)
    with h5py.File(f, 'r') as h5_file:
        feature = np.array(h5_file['feature'])
        feature = feature.squeeze()             # [batch,task,depth,n_feat]

    if use_sim:
        # Compute direction
        mod = np.linalg.norm(feature, ord=2, axis=2, keepdims=True)
        direc = feature/mod   # [batch,task,depth,n_feat]
        
        # Compute similarity
        sim_ = np.matmul(direc.transpose(0,1,3,2),direc) # [batch,task,n_feat,n_feat]
        sim += sim_.mean(axis=0)  # [task,n_feat,n_feat]
        
if use_sim:
    sim = sim/len(filesname)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(45,15)
    umb = int(n_feature/2)
    for t in range(n_task):
        axes[t].matshow(sim, cmap='coolwarm',vmin=-1,vmax=1,
                        extent=(0,n_feature,0,n_feature))
        axes[t].axis('off')
        for i in range(n_feature): axes[t].hlines(i,0,n_feature, lw=0.5,ls=':')
        axes[t].hlines(umb,0,n_feature)
        for i in range(n_feature): axes[t].vlines(i,0,n_feature, lw=0.5,ls=':')
        axes[t].vlines(umb,0,n_feature)
    plt.savefig('ftSim.svg',quality=900)
    
