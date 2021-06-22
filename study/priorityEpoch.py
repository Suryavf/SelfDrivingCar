import os
import glob

import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import common.prioritized
"""
Evolucion de priority en train
==============================
"""
dir_ = '/home/victor/Descargas/prior'
priorityFiles = glob.glob(os.path.join(dir_,'*.pck'))
priorityFiles.sort()

priList = list()
utcList = list()
cntList = list()

n_leaf    = None
n_samples = None
for path in priorityFiles:
    with open(path, 'rb') as handle:
        p = pickle.load(handle)

    if n_leaf is None:
        n_leaf = p['priority'].n_leaf

    priority      = p['priority']._data[n_leaf-1:]
    UTC           = p[     'UTC']._data[n_leaf-1:]
    sampleCounter = p['sampleCounter']

    if n_samples is None:
        n_samples = sampleCounter.shape[0]
    priority = priority[:n_samples]
    UTC      =      UTC[:n_samples]
    priority.sort()
    UTC.sort()
    
    priList.append(    priority   )
    utcList.append(      UTC      )
    cntList.append( sampleCounter )

priList = np.vstack(priList)
utcList = np.vstack(utcList)
cntList = np.vstack(cntList)

priList = np.log(priList)
priList = cv.resize(priList, dsize=(280,140))


vmax = np.max(priList)
vmin = np.min(priList)
vmin = vmin - (vmax-vmin)
plt.figure(figsize = (10,10))
plt.imshow(priList,cmap=cm.RdYlGn_r,vmin=vmin,vmax=vmax)
plt.show()
