import os
import glob
import h5py
import numpy as np
from   matplotlib import pyplot as plt

# Figure list
use_dist = True

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

# Initialize distribution
if use_dist:
    dist = np.zeros(n_task,n_feature,n_distr)

# Big loop
for n,f in enumerate(filesname):
    # Getting data
    print('Load file',f)
    with h5py.File(f, 'r') as h5_file:
        feature = np.array(h5_file['feature'])
        feature = feature.squeeze()             # [batch,task,depth,n_feat]
        feature = feature.transpose(1,0,3,2)    # [task,batch,n_feat,depth]
