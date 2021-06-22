import os
import glob
import h5py
import numpy as np
import cv2 as cv
from   matplotlib import pyplot as plt

# Parameters
path      = '/media/victor/Documentos/'
outpath   = '/media/victor/Documentos/Thesis/AttentionMap/Resume10'
dimImage  = ( 96,192)
dimEncode = ( 12, 24)
n_head    = 2
n_task    = 2
n_sample  = 120*20
def getint(name):
    basename = name.partition('.')
    _, num = basename[0].split('resume')
    return int(num)
filesname = glob.glob(os.path.join(path,'*.sy'))
filesname.sort()
filename  = '/media/victor/Documentos/resume10.sy'

# Getting data
with h5py.File(filename, 'r') as h5_file:
    image = np.array(h5_file['image']) 
    alpha = np.array(h5_file['alpha']) 
print('Load data done\n')


t = 0
# Genera mapas de atencion en video
alpha = alpha.reshape([n_sample,n_head,dimEncode[0],dimEncode[1],n_task])
alpha = alpha/alpha.max()

"""
for attmap,sample in zip(alpha,image):
    # a: [n_head,h,w,n_task] 
    # s: [3,H,W]

    tasks = list()
    for n in range(n_task):
        heads = list()
        for h in range(n_head):
            # Up-sampling
            att = attmap[h,:,:,n]
            att = cv.resize(att,None,fx=8,fy=8, interpolation = cv.INTER_AREA)
            att = cv.GaussianBlur(att,(11,11),0)
            #att = np.expand_dims(att,axis=0)    #  [1,H,W]

            # Apply
            sample = att #sample*att
            heads.append( sample ) #np.moveaxis(sample,0,2) )
        tasks.append( cv.vconcat(heads) )
    plt.figure(1); plt.clf()
    plt.imshow(cv.hconcat(tasks))
    plt.title('Frame ' + str(t))
    plt.pause(0.1)
    t += 1
"""
n_up = 8

for attmap,sample in zip(alpha,image):
    # a: [n_head,h,w,n_task] 
    # s: [3,H,W]
    sample = np.moveaxis(sample,0,2)
    maps = list()

    for n in range(n_task):
        map = np.zeros([ dimEncode[0]*n_up,dimEncode[1]*n_up ,3])
        for h in range(n_head):
            # Up-sampling
            att = attmap[h,:,:,n]
            att = cv.resize(att,None,fx=n_up,fy=n_up, interpolation = cv.INTER_AREA)
            att = cv.GaussianBlur(att,(11,11),0)
            map[:,:,h] = att
        maps.append( (0.5*sample+0.5*map) )
    
    img = cv.hconcat(maps)*255
    img = img.astype('float32')
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

    fileout = os.path.join(outpath,'sample%i.png'%t)
    cv.imwrite(fileout,img)

    #plt.imshow(cv.hconcat(maps))
    #fileout = os.path.join(outpath,'sample%i.png'%t)
    #plt.savefig(fileout)
    #print('Create %s'%fileout)
    #plt.title('Frame ' + str(t))
    #plt.pause(0.1)
    t += 1

