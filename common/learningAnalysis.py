import os
import glob
import h5py
import numpy  as np
import pandas as pd
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

pathTrain_tSNE = "../doc/test9.npy" # trainSeqTsne test
pathEval_tSNE  = "../doc/evalTsne.npy"



""" Ploting 2D-histogram of t-SNE
      * tsne : t-sne data      [n_samples,2]
      * focus: is it in focus? [n_samples]
"""
def plot_hist2D_tSNE(tsne,focus,xlim,ylim,
                    num=200,n_up=4,show=True,post=True,blur=False):
    # General
    hist = np.zeros([num,num])#,dtype=np.int)
    
    # Focus
    use_focus = (focus is not None)
    if use_focus: focs = np.zeros([num,num])#,dtype=np.int)

    xr = (xlim[1]-xlim[0])/num
    yr = (ylim[1]-ylim[0])/num
    
    # General
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        hist[nx,ny]+=1

        if use_focus: 
            if focus[k]: focs[nx,ny]+=1

    # Not use post-processing. Only count
    if not post: 
        if use_focus: return focs
        else        : return hist
        
    if use_focus:
        hist = hist + 1#0**-5
        hist = 255*focs/hist
        if n_up is not None:
            hist = cv.resize(hist,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
            if blur: hist = cv.GaussianBlur(hist,(5,5),0)
            hist = np.clip(hist,0,255)
    else:
        hist = 255*hist/hist.max()
        if n_up is not None:
            hist = cv.resize(hist,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
            if blur: hist = cv.GaussianBlur(hist,(5,5),0)
            hist = np.clip(hist,0,255)
            
    map = cv.applyColorMap(hist.astype((np.uint8)), cv.COLORMAP_TURBO) # COLORMAP_INFERNO COLORMAP_HOT COLORMAP_TURBO
    if show:
        canvas = cv.cvtColor(map, cv.COLOR_BGR2RGB)
        plt.imshow(canvas)
        plt.show()
    return map



def plot_tSNE(tsne,focus,path = None,
                         ax   = None, **kwargs):
    # Create canvas
    if ax is None: fig, ax = plt.subplots(figsize=(10,10))
    backgroundparams = {"alpha": kwargs.get("alpha", 0.1), "s": kwargs.get("s", 1)}
    foregroundparams = {"alpha": kwargs.get("alpha", 0.5), "s": kwargs.get("s", 5)}

    # Plotting background
    background = tsne[focus == 0]
    ax.scatter(background[:, 0],background[:, 1], c='#9aa191ff', rasterized=True, **backgroundparams)
    
    # Color focus
    foreground =  tsne[focus != 0]
    y          = focus[focus != 0]
    classes = np.unique(y)
    if len(classes) > 1:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
        point_colors = list(map(colors.get,y))
    else:
        point_colors = 'r'
    
    # Plotting focus
    ax.scatter(foreground[:, 0], foreground[:, 1], c=point_colors, rasterized=True, **foregroundparams)
    plt.show()

    # Save
    if path is None: plt.savefig('tsne.svg',quality=900,dpi=200)
    else           : plt.savefig(      path,quality=900,dpi=200)


def viewSubSetData(priority,min,max,num=200,n_up=4,train=True,show=True,post=True):
    
    if train: pathtsne = pathTrain_tSNE
    else    : pathtsne =  pathEval_tSNE
    
    with open(pathtsne, 'rb') as f:
        tsne = np.load(f)
    
    if tsne.shape[0] > len(priority):
        tsne = tsne[:len(priority),:]

    # Compute limits
    xrange = tsne[:,0].max() - tsne[:,0].min()
    yrange = tsne[:,1].max() - tsne[:,1].min()
    xlim = [tsne[:,0].min() - 0.05*xrange, tsne[:,0].max() + 0.05*xrange]
    ylim = [tsne[:,1].min() - 0.05*yrange, tsne[:,1].max() + 0.05*yrange]

    subset = (priority>min) & (priority<max)
    gh = plot_hist2D_tSNE(tsne,subset,xlim,ylim,num=num,n_up=n_up,show=show,post=post)

    if show: return None
    else   : return gh


def viewKim2017TrainDistribution(train=True):
    """
    c              0.0           1.4           3.0           5.0
    non = ["2009081241", "2009130017", "2009161542", "2009191117"]
    grd = ["2009061702", "2009041016", "2009020756", "2008311908"]
    fll = ["2105092139", "2009230924", "2105100106", "2105121334"]
    """
    model = 'Kim2017'
    limit = 121656
    ref_global = True

    namefile = 'priority.ps' if train else 'eval.er'

    def getData(path):
        with h5py.File(path, 'r') as h5file:
            if train:
                priority = np.array(h5file['PS.data'  ])
                n_leaf   =          h5file['PS.n_leaf'][()]
                priority = priority[n_leaf-1:limit+n_leaf-1]
            else:
                priority = np.array(h5file['loss'])
        return priority

    # No bias correcion without compensation
    non00 = getData(os.path.join('../aux/runs',model,'2009081241',namefile)).flatten()
    
    # Gradual bias correcion without compensation
    grd00 = getData(os.path.join('../aux/runs',model,'2009061702',namefile)).flatten()
    
    # Full bias correcion without compensation
    fll00 = getData(os.path.join('../aux/runs',model,'2105092139',namefile)).flatten()
    
    # No bias correcion with compensation
    non14 = getData(os.path.join('../aux/runs',model,'2009130017',namefile)).flatten()
    
    # Gradual bias correcion with compensation
    grd14 = getData(os.path.join('../aux/runs',model,'2009041016',namefile)).flatten()
    
    # Full bias correcion with compensation
    fll14 = getData(os.path.join('../aux/runs',model,'2009230924',namefile)).flatten()
    
    # Timeline
    n_time = 13 # 13
    n = 10**np.linspace(-4,0,n_time) # -3
    n = np.concatenate([[0],n])
    n1,n2 = n[:-1],n[1:]

    edge = 10
    n    = 100
    n_up = 4
    n_ln = n if ref_global else n_up*n

    if ref_global: canvas = np.zeros([8*edge + 6*n   , (n_time+1)*edge + n_time*n     ])
    else         : canvas = np.zeros([8*edge + 6*n_ln, (n_time+1)*edge + n_time*n_ln,3],dtype=np.uint8)
    
    idx = 0
    for a,b in zip(n1,n2):
        #b= 10
        # No bias correcion without compensation
        map = viewSubSetData(non00,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas[edge                :         edge+n_ln, 
               edge+idx*(edge+n_ln):(idx+1)*(edge+n_ln)] = map

        # Gradual bias correcion without compensation
        map = viewSubSetData(grd00,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas[edge+     edge+n_ln :     2 *(edge+n_ln), 
               edge+idx*(edge+n_ln):(idx+1)*(edge+n_ln)] = map

        # Full bias correcion without compensation
        map = viewSubSetData(fll00,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas[edge+  2*(edge+n_ln):     3 *(edge+n_ln), 
               edge+idx*(edge+n_ln):(idx+1)*(edge+n_ln)] = map
        

        # No bias correcion with compensation
        map = viewSubSetData(non14,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas[2*edge+  3*(edge+n_ln):edge+4 *(edge+n_ln), 
                 edge+idx*(edge+n_ln):(idx+1)*(edge+n_ln)] = map

        # Gradual bias correcion with compensation
        map = viewSubSetData(grd14,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas[2*edge+  4*(edge+n_ln):edge+5 *(edge+n_ln), 
                 edge+idx*(edge+n_ln):(idx+1)*(edge+n_ln)] = map

        # Full bias correcion with compensation
        map = viewSubSetData(fll14,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas[2*edge+  5*(edge+n_ln):edge+6 *(edge+n_ln), 
                 edge+idx*(edge+n_ln):(idx+1)*(edge+n_ln)] = map
        idx += 1

    if ref_global:
        p = canvas.flatten()
        p = np.percentile(p[p>0],99) 
        
        canvas = 255*canvas/p
        canvas = cv.resize(canvas,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
        canvas = np.clip(canvas,0,255)

        canvas = cv.applyColorMap( canvas.astype(np.uint8), cv.COLORMAP_TURBO) # COLORMAP_INFERNO COLORMAP_HOT COLORMAP_TURBO COLORMAP_JET COLORMAP_VIRIDIS

    canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
    plt.imsave('eval.png',canvas)
    
"""
code = '2009081241' # 2009230924 2107252143
model = 'Kim2017' # Kim2017 Approach
pathpry = os.path.join('../aux/runs',model,code,'priority.ps') # priority priority122
"""
viewKim2017TrainDistribution(False)


"""
pathtsne = pathTrain_tSNE

with open(pathtsne, 'rb') as f:
    tsne = np.load(f)

# Compute limits
xrange = tsne[:,0].max() - tsne[:,0].min()
yrange = tsne[:,1].max() - tsne[:,1].min()
xlim = [tsne[:,0].min() - 0.05*xrange, tsne[:,0].max() + 0.05*xrange]
ylim = [tsne[:,1].min() - 0.05*yrange, tsne[:,1].max() + 0.05*yrange]

plot_hist2D_tSNE(tsne,None,xlim,ylim,num=250,n_up=4,show=True)
"""