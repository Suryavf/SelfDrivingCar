import h5py
import numpy as np
import cv2 as cv
from   matplotlib import pyplot as plt

# Parameters
filename  = '/media/victor/Documentos/resume1.sy'
dimImage  = ( 96,192)
dimEncode = ( 12, 24)
n_head    = 2
n_task    = 3
n_sample  = 120*20

# Getting data
with h5py.File(filename, 'r') as h5_file:
    image = np.array(h5_file['image']) 
    alpha = np.array(h5_file['alpha']) 
print('Load data done\n')

# Histogram task
#task = 0
#plt.hist( alpha[:,:,:,:,task].reshape([n_sample*288*2]) ,bins=200)
#plt.show()

t = 0
# Genera mapas de atencion en video
alpha = alpha.reshape([n_sample,n_head,dimEncode[0],dimEncode[1],n_task])
alpha = alpha/alpha.max()
for attmap,sample in zip(alpha,image):
    # a: [n_head,h,w,n_task] 
    # s: [3,H,W]

    tasks = list()
    for n in range(n_task):
        heads = list()
        for h in range(n_head):
            # Up-sampling
            att = attmap[h,:,:,n]
            att = cv.resize(att,None,fx=8,fy=8, interpolation = cv.INTER_CUBIC)
            att = cv.GaussianBlur(att,(9,9),0)
            att = np.expand_dims(att,axis=0)    #  [1,H,W]

            # Apply
            sample = sample*att
            heads.append( np.moveaxis(sample,0,2) )
        tasks.append( cv.vconcat(heads) )
    plt.figure(1); plt.clf()
    plt.imshow(cv.hconcat(tasks))
    plt.title('Frame ' + str(t))
    plt.pause(0.1)
    t += 1
"""

# Getting data
with h5py.File(filename, 'r') as h5_file:
    state = np.array(h5_file['state']) 
    beta  = np.array(h5_file[ 'beta']) 

print(state.shape)
print(beta.shape)

"""