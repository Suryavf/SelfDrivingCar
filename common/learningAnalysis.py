import os
import glob
import h5py
import numpy  as np
import pandas as pd
import cv2 as cv
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# from scipy.integrate import interp2d

pathTrain_tSNE = "../doc/test9.npy" # trainSeqTsne test9 trainTsne1p100
pathEval_tSNE  = "../doc/evalTsne5.npy"
pathTrain_rdat = "../doc/trainDataRaw.npy"
pathEval_rdat  = "../doc/evalDataRaw.npy"

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


# -----------------------------------------------------------------------------------
def getData(path,train):
    limit = 121656      # Valor de sum-tree?
    with h5py.File(path, 'r') as h5file:
        if train:
            priority = np.array(h5file['PS.data'  ])
            n_leaf   =          h5file['PS.n_leaf'][()]
            priority = priority[n_leaf-1:limit+n_leaf-1]
        else:
            priority = np.array(h5file['loss'])
    return priority
def getMask(tsne,xlim,ylim,num):
    xr = (xlim[1]-xlim[0])/num
    yr = (ylim[1]-ylim[0])/num
    mask = np.zeros([num,num],dtype=np.bool)

    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        mask[nx,ny] =True
    return mask
def get_t_SNE(train):
    if train: pathtsne = pathTrain_tSNE
    else    : pathtsne =  pathEval_tSNE
    
    with open(pathtsne, 'rb') as f:
        tsne = np.load(f)
    
    # Compute limits
    xrange = tsne[:,0].max() - tsne[:,0].min()
    yrange = tsne[:,1].max() - tsne[:,1].min()
    xlim = [tsne[:,0].min() - 0.05*xrange, tsne[:,0].max() + 0.05*xrange]
    ylim = [tsne[:,1].min() - 0.05*yrange, tsne[:,1].max() + 0.05*yrange]
    return tsne,xlim,ylim
def getRawData(train,type=None):
    if train: pathraw = pathTrain_rdat
    else    : pathraw =  pathEval_rdat
    with open(pathraw, 'rb') as f:
        raw = np.load(f)
    if train:
        n   = raw.shape[0]
        raw = raw.reshape([ n,20,13  ])

    #   0   1   2    3   4  5  6  7             8             9      10     11   12
    # [st,thr,brk, vel, c1,c2,c3,c4, interOppLane,interSidewalk, cOther,cPedes,cCar]
    if   type == 'actions':
        if train: return raw[:,:,0:3]
        else    : return raw[ : ,0:3]
    elif type == 'velocity':
        if train: return raw[:,:,3]
        else    : return raw[ : ,3]
    elif type == 'command':
        if train: return raw[:,:,4:8]
        else    : return raw[ : ,4:8]
    elif type == 'interception':
        if train: return raw[:,:,8:10]
        else    : return raw[ : ,8:10]
    elif type == 'interceptionOppLane':
        if train: return raw[:,:,8]
        else    : return raw[ : ,8]
    elif type == 'interceptionSidewalk':
        if train: return raw[:,:,9]
        else    : return raw[ : ,9]
    elif type == 'collision':
        if train: return raw[:,:,10:]
        else    : return raw[ : ,10:]
    elif type == 'collisionOther':
        if train: return raw[:,:,10]
        else    : return raw[ : ,10]
    elif type == 'collisionPederatian':
        if train: return raw[:,:,11]
        else    : return raw[ : ,11]
    elif type == 'collisionCar':
        if train: return raw[:,:,12]
        else    : return raw[ : ,12]
    elif type == 'steering':
        if train: return raw[:,:,0]
        else    : return raw[ : ,0]
    elif type == 'throttle':
        if train: return raw[:,:,1]
        else    : return raw[ : ,1]
    elif type == 'brake':
        if train: return raw[:,:,2]
        else    : return raw[ : ,2]
    else: return raw

def BigViewKim2017DistributionComp(train=True):
    """
    c              0.0           1.4           3.0           5.0
    non = ["2009081241", "2009130017", "2009161542", "2009191117"]
    grd = ["2009061702", "2009041016", "2009020756", "2008311908"]
    fll = ["2105092139", "2009230924", "2105100106", "2105121334"]
    """
    model = 'Kim2017'
    ref_global = True

    namefile = 'priority.ps' if train else 'eval.er'

    # No bias correcion without compensation
    non00 = getData(os.path.join('../aux/runs',model,'2009081241',namefile),train).flatten()
    
    # Gradual bias correcion without compensation
    grd00 = getData(os.path.join('../aux/runs',model,'2009061702',namefile),train).flatten()
    
    # Full bias correcion without compensation
    fll00 = getData(os.path.join('../aux/runs',model,'2105092139',namefile),train).flatten()
    
    # No bias correcion with compensation
    non14 = getData(os.path.join('../aux/runs',model,'2009130017',namefile),train).flatten()
    
    # Gradual bias correcion with compensation
    grd14 = getData(os.path.join('../aux/runs',model,'2009041016',namefile),train).flatten()
    
    # Full bias correcion with compensation
    fll14 = getData(os.path.join('../aux/runs',model,'2009230924',namefile),train).flatten()
    
    # Timeline
    n_time = 13 # 13
    n = 10**np.linspace(-4,0,n_time) # -3
    n = np.concatenate([[0],n])
    n1,n2 = n[:-1],n[1:]

    edge = 10
    n    = 250
    n_up = 2
    n_ln = n if ref_global else n_up*n

    if ref_global: canvas = np.zeros([8*edge + 6*n   , (n_time+1)*edge + n_time*n     ])
    else         : canvas = np.zeros([8*edge + 6*n_ln, (n_time+1)*edge + n_time*n_ln,3],dtype=np.uint8)
    background = np.zeros([8*edge + 6*n, (n_time+1)*edge + n_time*n ],dtype=np.uint8) # 680, 1440   2720, 5760
    
    idx = 0
    tsne,xlim,ylim = get_t_SNE(train)
    mask = getMask(tsne,xlim,ylim,n)
    for a,b in zip(n1,n2):
        #b= 10
        # No bias correcion without compensation
        x1,x2 = edge                ,         edge+n_ln 
        y1,y2 = edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
        map = viewSubSetData(non00,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas    [x1:x2,y1:y2] = map
        background[x1:x2,y1:y2] = mask

        # Gradual bias correcion without compensation
        x1,x2 = edge+     edge+n_ln ,     2 *(edge+n_ln)
        y1,y2 = edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
        map = viewSubSetData(grd00,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas    [x1:x2,y1:y2] = map
        background[x1:x2,y1:y2] = mask

        # Full bias correcion without compensation
        x1,x2 = edge+  2*(edge+n_ln),     3 *(edge+n_ln)
        y1,y2 = edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
        map = viewSubSetData(fll00,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas    [x1:x2,y1:y2] = map
        background[x1:x2,y1:y2] = mask
        

        # No bias correcion with compensation
        x1,x2 = 2*edge+  3*(edge+n_ln),edge+4 *(edge+n_ln)
        y1,y2 =   edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
        map = viewSubSetData(non14,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas    [x1:x2,y1:y2] = map
        background[x1:x2,y1:y2] = mask

        # Gradual bias correcion with compensation
        x1,x2 = 2*edge+  4*(edge+n_ln),edge+5 *(edge+n_ln)
        y1,y2 =   edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
        map = viewSubSetData(grd14,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas    [x1:x2,y1:y2] = map
        background[x1:x2,y1:y2] = mask

        # Full bias correcion with compensation
        x1,x2 = 2*edge+  5*(edge+n_ln),edge+6 *(edge+n_ln)
        y1,y2 =   edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
        map = viewSubSetData(fll14,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        canvas    [x1:x2,y1:y2] = map
        background[x1:x2,y1:y2] = mask
        idx += 1

    if ref_global:
        p = canvas.flatten()
        p = np.percentile(p[p>0],99) 
        
        canvas = 255*canvas/p
        canvas = cv.resize(canvas,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
        canvas = np.clip(canvas,0,255)
        canvas = cv.applyColorMap( canvas.astype(np.uint8), cv.COLORMAP_TURBO) # COLORMAP_INFERNO COLORMAP_HOT COLORMAP_TURBO COLORMAP_JET COLORMAP_VIRIDIS

        background = cv.resize(background,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)

    canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
    canvas = canvas*np.expand_dims(background,axis=2)
    canvas[canvas==0] = 5
    plt.imsave('train.png',canvas)
    

def viewKim2017DistributionComp(train=True,onecanvas=True):
    #         PS        PS+BC    PS+BC+UCT
    #'2009081241','2105092139','2009230924'
    model = 'Kim2017'
    limit = 121656
    ref_global = True

    namefile = 'priority.ps' if train else 'eval.er'

    # No bias correcion without compensation
    ps1 = getData(os.path.join('../aux/runs',model,'2009081241',namefile),train).flatten()
    
    # Gradual bias correcion without compensation
    ps2 = getData(os.path.join('../aux/runs',model,'2105092139',namefile),train).flatten()
    
    # Full bias correcion without compensation
    ps3 = getData(os.path.join('../aux/runs',model,'2009230924',namefile),train).flatten()
    
    
    # Timeline
    n_time = 9 # 13
    n = [  0, 
          10**-4, 3.16227766*10**-4,
          10**-3, 3.16227766*10**-3,
          10**-2, 3.16227766*10**-2,
          10**-1, 3.16227766*10**-1,
          1]
    n1,n2 = n[:-1],n[1:]

    edge = 10
    if train: n,n_up = 500,1
    else    : n,n_up = 250,2
    n_ln = n if ref_global else n_up*n

    if onecanvas:
        if ref_global: canvas = np.zeros([4*edge + 3*n   , (n_time+1)*edge + n_time*n     ])
        else         : canvas = np.zeros([4*edge + 3*n_ln, (n_time+1)*edge + n_time*n_ln,3],dtype=np.uint8)
        background = np.zeros([4*edge + 3*n, (n_time+1)*edge + n_time*n ],dtype=np.uint8) # 680, 1440   2720, 5760
    else:
        # Save maps
        P1,P2,P3 = [],[],[]
    countP1,countP2,countP3 = [], [], []

    idx = 0
    tsne,xlim,ylim = get_t_SNE(train)
    mask = getMask(tsne,xlim,ylim,n)
    for a,b in zip(n1,n2):
        #b= 10
        # No bias correcion without compensation
        map = viewSubSetData(ps1,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        if onecanvas:
            x1,x2 = edge                ,         edge+n_ln 
            y1,y2 = edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
            canvas    [x1:x2,y1:y2] = map
            background[x1:x2,y1:y2] = mask
        else: P1.append(map)
        countP1.append(map.sum())   # For metrics

        # Gradual bias correcion without compensation
        map = viewSubSetData(ps2,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        if onecanvas:
            x1,x2 = edge+     edge+n_ln ,     2 *(edge+n_ln)
            y1,y2 = edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
            canvas    [x1:x2,y1:y2] = map
            background[x1:x2,y1:y2] = mask
        else: P2.append(map)
        countP2.append(map.sum())   # For metrics

        # Full bias correcion without compensation
        map = viewSubSetData(ps3,a,b,num=n,n_up=n_up,train=train,show=False,post=not ref_global)
        if onecanvas:
            x1,x2 = edge+  2*(edge+n_ln),     3 *(edge+n_ln)
            y1,y2 = edge+idx*(edge+n_ln),(idx+1)*(edge+n_ln)
            canvas    [x1:x2,y1:y2] = map
            background[x1:x2,y1:y2] = mask
        else: P3.append(map)
        countP3.append(map.sum())   # For metrics
        idx += 1

    if ref_global:
        if onecanvas:
            p = canvas.flatten()
            p = np.percentile(p[p>0],99) 

            canvas = 255*canvas/p
            canvas = cv.resize(canvas,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
            canvas = np.clip(canvas,0,255)
            canvas = cv.applyColorMap( canvas.astype(np.uint8), cv.COLORMAP_TURBO) # COLORMAP_INFERNO COLORMAP_HOT COLORMAP_TURBO COLORMAP_JET COLORMAP_VIRIDIS

            background = cv.resize(background,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
        else:
            p = np.hstack([np.hstack(P1),np.hstack(P2),np.hstack(P3)]).flatten()
            p = np.percentile(p[p>0],99)
            for i in range(len(P1)):
                P1[i] = cv.resize(255*P1[i]/p,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
                P1[i] = np.clip(P1[i],0,255).astype(np.uint8)
                P1[i] = cv.applyColorMap( P1[i], cv.COLORMAP_TURBO)

                P2[i] = cv.resize(255*P2[i]/p,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
                P2[i] = np.clip(P2[i],0,255).astype(np.uint8)
                P2[i] = cv.applyColorMap( P2[i], cv.COLORMAP_TURBO)

                P3[i] = cv.resize(255*P3[i]/p,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
                P3[i] = np.clip(P3[i],0,255).astype(np.uint8)
                P3[i] = cv.applyColorMap( P3[i], cv.COLORMAP_TURBO)

    # Print metrics
    countP1 = np.array(countP1)
    countP2 = np.array(countP2)
    countP3 = np.array(countP3)

    print('Porcentages')
    print( 100*countP1/countP1.sum() )
    print( 100*countP2/countP2.sum() )
    print( 100*countP3/countP3.sum() )

    # Save
    if onecanvas:
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = canvas*np.expand_dims(background,axis=2)
        canvas[canvas==0] = 5
        plt.imsave('eval.png',canvas)
    else: 
        mask = mask.astype(np.uint8)
        mask = cv.resize(mask,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
        for i in range(len(P1)):
            P1[i] = cv.cvtColor(P1[i], cv.COLOR_BGR2RGB)*np.expand_dims(mask,axis=2)
            P2[i] = cv.cvtColor(P2[i], cv.COLOR_BGR2RGB)*np.expand_dims(mask,axis=2)
            P3[i] = cv.cvtColor(P3[i], cv.COLOR_BGR2RGB)*np.expand_dims(mask,axis=2)

            P1[i][ P1[i]==0 ] = 5
            P2[i][ P2[i]==0 ] = 5
            P3[i][ P3[i]==0 ] = 5
            
            plt.imsave('out/NBC%i.png'%(i+1),P1[i])
            plt.imsave('out/GBC%i.png'%(i+1),P2[i])
            plt.imsave('out/FBC%i.png'%(i+1),P3[i])


# ----------------------------------
def viewKim2017Distribution(train=True):
    # Parameters
    if train: n,n_up = 600,1
    else    : n,n_up = 300,2
    
    # Get t-SNE and Mask
    tsne,xlim,ylim = get_t_SNE(train)
    mask = getMask(tsne,xlim,ylim,n)
    mask = mask.astype(np.uint8)
    mask = cv.resize(mask,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    mask = np.expand_dims(mask,axis=2)

    xr = (xlim[1]-xlim[0])/n
    yr = (ylim[1]-ylim[0])/n

    # DENSITY
    # --------------------------------------------------------------------------------
    
    # Getting map
    map = plot_hist2D_tSNE(tsne,None,xlim,ylim,num=n,n_up=n_up,show=False,post=False)

    # Normalization
    print("Density max", map.max())
    map = 255*map/106 #map.max()

    # Coloring
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)
    map = cv.applyColorMap( map.astype(np.uint8), cv.COLORMAP_TURBO)

    # Masking
    map = map*mask
    map[map==0] = 5
    
    # Save
    map = cv.cvtColor(map, cv.COLOR_BGR2RGB)
    plt.imsave('density.png',map)


    # COMMAND
    # --------------------------------------------------------------------------------
    # 0 Follow lane, 1 Left, 2 Right, 3 Straight

    #                     R   G   B
    color = np.array([ [250,146,  0],   # Follow lane   #FA9200
                       [255,  0,255],   # Turn left     #FF00FF
                       [103,250, 25],   # Turn right    #67FA19
                       [ 25,163,250] ]) # Straight      #19A3FA
    dat = getRawData(train,'command')
    
    # Map
    map  = np.zeros([n,n,3])
    hist = np.zeros([n,n,1])
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        
        # Get color for command
        cmd = dat[k]  # [20,4]
        # print(cmd)
        if not train: cmd = cmd.reshape([1,4])
        c = np.matmul( cmd,color )
        if train: c = c.sum(0)#.flatten()
        else    : c = c[0]
        
        map[nx,ny] += c
        
        # Update count
        hist[nx,ny]+=1
    hist = hist + 1 # Anti NaNs
    if train: map = map/(hist*20) 
    else    : map = map/ hist

    map = map.astype(np.uint8)
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)

    # Save
    plt.imsave('command.png',map)



    # VELOCITY
    # --------------------------------------------------------------------------------
    dat = getRawData(train,'velocity')

    # Map
    map  = np.zeros([n,n])
    hist = np.zeros([n,n])
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        
        # Get color for command
        if train: map[nx,ny] += dat[k].mean()
        else    : map[nx,ny] += dat[k]

        # Update count
        hist[nx,ny]+=1
    hist = hist + 1 # Anti NaNs
    map = map/hist
    print('Velocity max:',map.max())
    map = 255*map

    # Coloring
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)
    map = cv.applyColorMap( map.astype(np.uint8), cv.COLORMAP_TURBO)

    # Masking
    map = map*mask
    map[map==0] = 5
    
    # Save
    map = cv.cvtColor(map, cv.COLOR_BGR2RGB)
    plt.imsave('velocity.png',map)




    # FAIL: interception, collision
    # --------------------------------------------------------------------------------
    #                     R   G   B
    color = np.array([ [  0,255,  0],   # Interception  
                       [255,  0,  0] ]) # Collision     
    inter = getRawData(train,'interception')
    colls = getRawData(train,'collision')
    inter = inter.sum(-1,keepdims=True)   # All cases
    colls = colls.sum(-1,keepdims=True)
    
    # Threshold
    inter = inter>0.05
    colls = colls>0
    if train:
        inter = inter.mean(1)    # Mean in sequence
        colls = colls.mean(1)
    fail = np.hstack([inter,colls])
    
    # Only for fails
    if train: n,n_up = 150,4
    else    : n,n_up = 150,4
    xr = (xlim[1]-xlim[0])/n
    yr = (ylim[1]-ylim[0])/n

    # Map
    map  = np.zeros([n,n,3])
    hist = np.zeros([n,n,1])
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        
        # Get color for command
        val = fail[k]  # [2,]
        c = np.matmul( val,color )
        map[nx,ny] += c
        
        # Update count
        if val.sum()>0:
            hist[nx,ny]+=1
    hist = hist + 1 
    map = map/hist
    map = map.astype(np.uint8)
    map[map==0] = 20
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)

    # Masking
    map = map*mask
    map[map==0] = 5

    # Save
    plt.imsave('fail.png',map)


    # Steering angle
    # --------------------------------------------------------------------------------
    dat = getRawData(train,'steering')
    dat = (dat+1)/2

    # Map
    map  = np.zeros([n,n])
    hist = np.zeros([n,n])
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        
        # Get color for command
        if train: map[nx,ny] += dat[k].mean()
        else    : map[nx,ny] += dat[k]

        # Update count
        hist[nx,ny]+=1
    hist = hist + 1 # Anti NaNs
    map = map/hist
    print('Steering max:',map.max())
    map = 255*map

    # Coloring
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)
    map = cv.applyColorMap( map.astype(np.uint8), cv.COLORMAP_TURBO)

    # Masking
    map = map*mask
    map[map==0] = 5
    
    # Save
    map = cv.cvtColor(map, cv.COLOR_BGR2RGB)
    plt.imsave('steering.png',map)



    # Throttle 
    # --------------------------------------------------------------------------------
    dat = getRawData(train,'throttle')

    # Map
    map  = np.zeros([n,n])
    hist = np.zeros([n,n])
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        
        # Get color for command
        if train: map[nx,ny] += dat[k].mean()
        else    : map[nx,ny] += dat[k]

        # Update count
        hist[nx,ny]+=1
    hist = hist + 1 # Anti NaNs
    map = map/hist
    print('Throttle max:',map.max())
    map = 255*map

    # Coloring
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)
    map = cv.applyColorMap( map.astype(np.uint8), cv.COLORMAP_TURBO)

    # Masking
    map = map*mask
    map[map==0] = 5
    
    # Save
    map = cv.cvtColor(map, cv.COLOR_BGR2RGB)
    plt.imsave('throttle.png',map)
    


    # Brake 
    # --------------------------------------------------------------------------------
    dat = getRawData(train,'brake')  # brake collisionPederatian collisionCar collisionOther interceptionOppLane interceptionSidewalk

    # Map
    map  = np.zeros([n,n])
    hist = np.zeros([n,n])
    for k in range(len(tsne)):
        val = tsne[k]
        nx = int( (val[0]-xlim[0])/xr )
        ny = int( (val[1]-ylim[0])/yr )
        
        # Get color for command
        if train: map[nx,ny] += dat[k].mean()
        else    : map[nx,ny] += dat[k]

        # Update count
        hist[nx,ny]+=1
    hist = hist + 1 # Anti NaNs
    map = map/hist
    print('Brake max:',map.max())
    map = 255*map

    # Coloring
    map = cv.resize(map,None,fx=n_up,fy=n_up, interpolation = cv.INTER_CUBIC)
    map = np.clip(map,0,255)
    map = cv.applyColorMap( map.astype(np.uint8), cv.COLORMAP_TURBO)

    # Masking
    map = map*mask
    map[map==0] = 5
    
    # Save
    map = cv.cvtColor(map, cv.COLOR_BGR2RGB)
    plt.imsave('brake.png',map)



# ----------------------------------
def viewKim2017Cases(train=True):
    #         PS        PS+BC    PS+BC+UCT
    #'2009081241','2105092139','2009230924'
    model = 'Kim2017'
    namefile = 'priority.ps' if train else 'eval.er'
    
    # No bias correcion without compensation
    ps1 = getData(os.path.join('../aux/runs',model,'2009081241',namefile),train).flatten()
    
    # Gradual bias correcion without compensation
    ps2 = getData(os.path.join('../aux/runs',model,'2105092139',namefile),train).flatten()
    
    # Full bias correcion without compensation
    ps3 = getData(os.path.join('../aux/runs',model,'2009230924',namefile),train).flatten()

    ps1 = np.log10(ps1)
    ps2 = np.log10(ps2)
    ps3 = np.log10(ps3)

    str = getRawData(train,'steering')
    thr = getRawData(train,'throttle')
    brk = getRawData(train,'brake')
    vel = getRawData(train,'velocity')

    cmd = getRawData(train,'command')
    ntr = getRawData(train,'interception')
    col = getRawData(train,'collision')

    # Case 1: collision 
    # --------------------------------------------------------------------------------
    cnd = col.sum(1).sum(1)>1
    print('Case 1',cnd.sum())
    #sns.set_style('whitegrid')
    sns.set(rc={'figure.figsize':(10,2)})
    sns.kdeplot( ps1[cnd],label='PS'       ,bw_adjust=0.3)
    sns.kdeplot( ps2[cnd],label='PS+BC'    ,bw_adjust=0.3)
    sns.kdeplot( ps3[cnd],label='PS+BC+UCT',bw_adjust=0.3)
    plt.legend()
    plt.title('Collision')
    plt.xlim([-4,0])
    plt.grid()
    plt.show()
    
    # Case 2: stop condition 
    # --------------------------------------------------------------------------------
    c1 = vel.mean(1)<1/90 
    c2 = thr.mean(1)>0.8
    cnd = c1*c2
    print('Case 2',cnd.sum())
    sns.kdeplot( ps1[cnd],label='PS'       ,bw_adjust=0.3)
    sns.kdeplot( ps2[cnd],label='PS+BC'    ,bw_adjust=0.3)
    sns.kdeplot( ps3[cnd],label='PS+BC+UCT',bw_adjust=0.3)
    plt.legend()
    plt.title('Stop condition')
    plt.xlim([-4,0])
    plt.show()

    # Case 3: easy condition
    # --------------------------------------------------------------------------------
    c1 = vel.mean(1)> 0.17
    c2 = vel.mean(1)< 0.25
    c3 = str.mean(1)< 0.1
    c4 = str.mean(1)>-0.1
    c5 = thr.mean(1)<0.51
    c6 = brk.mean(1)<0.01
    c7 = col.sum(1).sum(1)<0.1
    c8 = ntr.sum(1).sum(1)<0.1
    cnd = c1*c2*c3*c4*c5*c6*c7*c8
    print('Case 3',cnd.sum())
    sns.kdeplot( ps1[cnd],label='PS'       ,bw_adjust=0.3)
    sns.kdeplot( ps2[cnd],label='PS+BC'    ,bw_adjust=0.3)
    sns.kdeplot( ps3[cnd],label='PS+BC+UCT',bw_adjust=0.3)
    plt.legend()
    plt.title('Easy condition')
    plt.xlim([-4,0])
    plt.show()

    # Case 4: hard steering
    # --------------------------------------------------------------------------------
    c1 = str.mean(1)> 0.4
    c2 = str.mean(1)<-0.4
    cnd = c1 + c2
    print('Case 4',cnd.sum())
    sns.kdeplot( ps1[cnd],label='PS'       ,bw_adjust=0.3)
    sns.kdeplot( ps2[cnd],label='PS+BC'    ,bw_adjust=0.3)
    sns.kdeplot( ps3[cnd],label='PS+BC+UCT',bw_adjust=0.3)
    plt.legend()
    plt.title('Hard steering')
    plt.xlim([-4,0])
    plt.show()
    

#viewKim2017DistributionComp(False,False)
#viewKim2017Distribution(True)
viewKim2017Cases(True)