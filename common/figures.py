import os
from   os      import listdir
from   os.path import isfile, join
import datetime

from IPython.core.debugger import set_trace

import numpy as np
import matplotlib
import cv2 as cv
matplotlib.use('Agg')
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.animation as animation 
from scipy.interpolate import interp2d
from random import shuffle


""" Save plot
    ---------
    Args:
        data: Data to plot
        path: File to save

    Return: files list
"""
def saveScatterSteerSpeed(steer,speed,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            c = cmd[idx]
            axs[i,j].scatter(steer[command==c],speed[command==c],alpha=0.1)
            axs[i,j].grid(True)
            axs[i,j].set_xlabel("Steer")
            axs[i,j].set_ylabel("Speed")
            axs[i,j].set_title(hgl[idx])
            axs[i,j].set_xlim(-1.2,1.2)
            axs[i,j].set_ylim( -20, 90)
            idx += 1
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    fig.savefig(path)
    plt.close('all')
def saveScatterError(steer,steerErr,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            c = cmd[idx]
            axs[i,j].scatter(steer[command==c],steerErr[command==c],alpha=0.1)
            axs[i,j].grid(True)
            axs[i,j].set_xlabel("Steer (True)")
            axs[i,j].set_ylabel("Steer Error")
            axs[i,j].set_title(hgl[idx])
            axs[i,j].set_xlim(-1.2,1.2)
            axs[i,j].set_ylim(-0.1,1.2)
            idx += 1
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    fig.savefig(path)
    plt.close('all')
def saveScatterPolarSteerSpeed(steer,speed,command,path):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    idx = 0

    fig, axs = plt.subplots(2, 2)
    steer = steer.reshape(-1)
    speed = speed.reshape(-1)

    x = speed*np.cos(steer)
    y = speed*np.sin(steer)
    for i in range(2):
        for j in range(2):
            c = cmd[idx]
            axs[i,j].scatter(y[command==c],x[command==c],alpha=0.05)
            axs[i,j].grid(True)
            axs[i,j].set_xlabel("y")
            axs[i,j].set_ylabel("x")
            axs[i,j].set_title(hgl[idx])
            axs[i,j].set_ylim(-20,90)
            axs[i,j].set_xlim(-50,50)
            idx += 1
    fig.tight_layout()
    fig.set_size_inches(10,10)
    fig.savefig(path)
    plt.close('all')

""" Save histogram
    --------------
    Args:
        data: Data to histogram
        path: File to save

    Return: files list
"""
def saveHistogramSteerSpeed(steer,speed,path):
    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)

    axs[0].hist(x=steer, bins=180)
    axs[0].set_xlim(-1.3,1.3)
    axs[0].set_ylim(0,18000)
    axs[0].set_title("Steer")

    axs[1].hist(x=speed, bins=180)
    axs[1].set_xlim(-5,90)
    axs[1].set_ylim(0,15000)
    axs[1].set_title("Speed")

    fig.set_size_inches(12,10)
    fig.savefig(path)
    plt.close('all')

def saveHistogramSteer(steer,path):
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    axs.hist(x=steer, bins=180,range=(-1.3,1.3))
    axs.set_xlim(-1.3,1.3)
    axs.set_ylim(0,18000)
    axs.set_title("Steer")

    fig.set_size_inches(12,5)
    fig.savefig(path)
    plt.close('all')


class savePlotByStep():
    def __init__(self, path,name):
        self._path = path
        self._name = name
        self._values = list()

    def update(self,values):
        # Add values 
        self._values.append(values)
        n_epoch = len(self._values)

        if n_epoch > 2:
            epochs = list( range(1,n_epoch+1) )
            name = self._name + ".svg"
            path = join(self._path,name)
            
            fig, ax = plt.subplots()
            ax.plot(epochs,self._values)
            plt.xlabel('Epoch')
            plt.ylabel(self._name)
            plt.xlim(1,n_epoch)
            plt.savefig(path)
            plt.close('all')

    def reset(self):
        self._values = list()

class save2PlotByStep():
    def __init__(self, path,name,line1,line2):
        self._path = path
        self._name = name
        self._line1 = line1
        self._line2 = line2
        self._values1 = list()
        self._values2 = list()

    def update(self,val1,val2):
        # Add values 
        self._values1.append(val1)
        self._values2.append(val2)
        n_epoch = len(self._values1)

        if n_epoch > 2:
            epochs = list( range(1,n_epoch+1) )
            name = self._name + ".svg"
            path = join(self._path,name)
            
            fig, ax = plt.subplots()
            ax.plot(epochs,self._values1)
            ax.plot(epochs,self._values2)

            plt.legend([self._line1,self._line2])
            plt.xlabel("Epoch")
            plt.savefig(path)
            plt.close('all')

    def reset(self):
        self._values1 = list()
        self._values2 = list()
        

""" Save color mersh
    ----------------
    Args:
        data: Data to histogram
        path: File to save

    Return: files list
"""
def saveColorMershError(msr,msrErr,command,path,dom=(-1.20, 1.20),resolution=30,name='Steer'):
    hgl = ['Follow lane','Left Turn','Straight','Right Turn']
    cmd = [0,1,3,2]
    
    # msr       (74800,)
    # msrErr    (74800,)
    # command   (74800,)
    
    if dom[0] == 0: mirror = False
    else          : mirror = True
    
    # Length  
    if mirror: x_len,y_len = (2*resolution,resolution)
    else     : x_len,y_len = (  resolution,resolution)
    
    x_min,x_max =   dom
    y_min,y_max = (-dom[1]/100, dom[1])

    x    = np.linspace(x_min,x_max,x_len )
    y    = np.linspace(y_min,y_max,y_len )
    xnew = np.linspace(x_min,x_max,  800 )
    ynew = np.linspace(y_min,y_max,  800 )

    e = 0.0001
    x_cte = (x_max-x_min)/(x_len) + e

    idx = 0
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            # Select command
            st  = msr   [command==cmd[idx]]
            err = msrErr[command==cmd[idx]]

            mersh = list()
            for _ in range(x_len): mersh.append( list() )
            for s,e in zip(st,err):
                pst = int(np.floor( (s-x_min)/x_cte ))
                if pst < x_len:
                    mersh[pst].append( e )
            
            # Create mersh
            for k in range(x_len):
                vec = mersh[k]
                if len(vec)>2:
                    # Coefficient (values in plot range [0,1.2])
                    coeff = np.sum( np.array(vec)<=x_max ) / len(vec)

                    # Getting histogram
                    vec , _ = np.histogram( vec, y_len,range=(y_min,y_max) )
                    maxVec  = np.max( vec )

                    if maxVec < 1:
                        mersh[k] = np.zeros(y_len)
                    else:
                        # Remove noise
                        vec     = np.multiply( vec,(vec>1))
                        # Normalization (respect to the maximum)
                        mersh[k] = vec / maxVec
                        # Normalization (respect to data in plot range)
                        mersh[k] = coeff*mersh[k]
                else:
                    mersh[k] = np.zeros(y_len)
            mersh = np.array( mersh )
            
            # Smooth mersh
            f = interp2d(x, y, mersh.T, kind='linear')
            mersh = f(xnew,ynew)
            mersh = np.multiply(mersh , (mersh>0.01))
            mersh = np.multiply(mersh , (mersh<  1 )) + np.multiply(np.ones(mersh.shape),(mersh>=1))
            Xn, Yn = np.meshgrid(xnew, ynew)
            
            axs[i,j].pcolormesh(Xn, Yn, mersh, cmap='jet', vmin=0, vmax=1, shading='auto')
            axs[i,j].set_xlabel(name+" (True)")
            axs[i,j].set_ylabel(name+" Error")
            axs[i,j].set_title(hgl[idx])
            idx += 1 

    fig.tight_layout()
    fig.set_size_inches(10,6.7)
    fig.savefig(path)
    plt.close('all')
    

def saveHistAnimation(data):
    def update_hist(num): 
        plt.cla() 
        plt.hist(data[num],bins=500) 
        plt.ylim(0,2800) 

    fig = plt.figure() 
    ani = animation.FuncAnimation(fig, update_hist, data.shape[0], fargs=(data, ) ) 
    ani.save("samples.mp4")  
    


def saveHeatmap(orig,pred,label,path,range):
    # Parameters
    n    = 8
    bins = 30
    
    # Figure
    fig = plt.figure(figsize=(5,5))
    gs  = gridspec.GridSpec(n,n)
    
    # Plot regions
    figScatter  = fig.add_subplot(gs[1:n,:n-1])
    figMargOrig = fig.add_subplot(gs[ 0 ,:n-1],sharex=figScatter)
    figMargPred = fig.add_subplot(gs[1:n, n-1],sharey=figScatter)
    # Range
    range[0] -= 0.15
    range[1] += 0.15

    # Heatmap
    heatmap, xedges, yedges = np.histogram2d(orig, pred, bins=bins, range=[range,range])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = np.log( 120 + heatmap.T )
    heatmap = cv.pyrUp(heatmap)
    heatmap = cv.pyrUp(heatmap)
    heatmap = cv.pyrUp(heatmap)
    heatmap = cv.pyrUp(heatmap)
    heatmap = cv.pyrUp(heatmap)
    
    figScatter.imshow(heatmap, extent=extent, origin='lower', cmap='GnBu') # BuPu GnBu PuBu binary bone gist_earth gray_r inferno_r nipy_spectral_r rainbow
    n_orig, _, _ = figMargOrig.hist(orig, bins=bins, density=True)
    n_pred, _, _ = figMargPred.hist(pred, bins=bins, density=True, orientation="horizontal")
    
    figScatter .set_xlim(range[0],range[1])
    figScatter .set_ylim(range[0],range[1])
    figMargOrig.set_xlim(range[0],range[1])
    figMargOrig.set_ylim( 0,np.max(n_orig))
    figMargPred.set_ylim(range[0],range[1])
    figMargPred.set_xlim( 0,np.max(n_pred))

    # Turn off tick labels on marginals
    plt.setp(figMargOrig.get_xticklabels(), visible=False)
    plt.setp(figMargOrig.get_yticklabels(), visible=False)
    plt.setp(figMargPred.get_xticklabels(), visible=False)
    plt.setp(figMargPred.get_yticklabels(), visible=False)

    # Set labels on joint
    figScatter.set_xlabel('Original '  + label)
    figScatter.set_ylabel('Predicted ' + label)

    # Set labels on marginals
    #figMargOrig.set_ylabel('Original '  + label)
    #figMargPred.set_xlabel('Predicted ' + label)

    fig.savefig(path)
    plt.close('all')
    
