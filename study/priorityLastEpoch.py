import os
import glob

import pickle
from re import S
import cv2 as cv
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import common.prioritized
from matplotlib.colors import LogNorm

# 2009081241 2009130017 2009161542 2009191117
# 2009061702 2009041016 2009020756 2008311908
#            2009230924 
#model = 'Kim2017'
#code  = '2009130017'
#dir_  = '/media/victor/Documentos/Thesis/Priority/'

#path = os.path.join(dir_,model,code,'priority.pck')

class Master(object):
    """ Constructor """
    def __init__(self,model,codes,dir_=''):
        # Parameters
        self.model = model
        self.codes = codes
        self.dir_  = dir_


    """ Load datas """
    def load(self,path):
        df = {}
        with open(path, 'rb') as handle:
            db = pickle.load(handle)
            if 'priority' in db:
                n_leaf        = db['priority'].n_leaf
                priority      = db['priority']._data[n_leaf-1:]

            if 'UTC' in db:
                sampleCounter = db['sampleCounter']
                n_samples     = sampleCounter.shape[0]
                UTC           = db[     'UTC']._data[n_leaf-1:]
                
                df['UTC'] = UTC[:n_samples]
            else:
                n_samples = np.where( priority==0 )[0][0]
            df['priority'] = priority[:n_samples]
        return df
    
    """ Figure 01 """
    def fig01(self):
        # Loop
        for code in self.codes:
            path = os.path.join(self.dir_,self.model,code,'priority.pck')
            df   = self.load(path)

            # Figure
            plt.hist2d(df['priority'], df['UTC'], bins=100, norm=LogNorm(), 
                                                  range=[[0.00045,0.48751],
                                                         [0.70667,1.25366]] )#, cmap='Blues')
            plt.xlabel('Priority')
            plt.ylabel('UTC')
            plt.title('Priority vs UTC')
            cb = plt.colorbar()
            cb.set_label('counts in bin')
            plt.show()

    """ Figure 02 """
    def fig02(self,useLog = True,show=True,name=''):
        x = np.linspace(0,100,1000)[1:]
        plt.figure(figsize=(8,3))

        # Loop
        for code in self.codes:
            path = os.path.join(self.dir_,self.model,code,'priority.pck')
            df   = self.load(path)

            # Compute percentile
            p = list()
            priority =  df['priority']
            for i in x:
                p.append( np.percentile(priority, i) )
            
            # Plot
            plt.plot( np.array(p),x )
            plt.vlines( np.median(priority),ymin=1,ymax=100,colors='r', ls=':', lw=1)
        plt.xlabel('Loss value')
        plt.ylabel('Percentile')
        plt.hlines( 50,xmin=0.001,xmax=1,colors='r', ls=':', lw=1)
        plt.xlim([0.001,1])
        plt.ylim([1,100])
        if useLog: plt.xscale('log')
        plt.grid(True)
        if show: plt.show()
        else   : plt.savefig('perLog%s.svg'%name,quality=900)

    """ Figure 03 """
    def fig03(self,useLog = True,show=True,name=''):
        # Parameters
        xmin,xmax = 10**-4,  0.5
        ymin,ymax =    0  , 1100
        n    = 3000
        x = np.linspace(xmin,xmax,n)
        plt.figure(figsize=(8,2))

        # Loop
        for code in self.codes:
            path = os.path.join(self.dir_,self.model,code,'priority.pck')
            df   = self.load(path)
            
            # Compute percentile
            hist,_ = np.histogram(df['priority'],bins=n,range=[xmin,xmax])
            plt.plot( x,hist )
            plt.vlines( np.median(df['priority']),ymin=ymin,ymax=ymax,colors='r', ls=':', lw=1)
            print(np.median(df['priority']))
        plt.xlabel('Loss value')
        plt.ylabel('Frequency')
        if useLog: plt.xscale('log')
        plt.legend(self.codes)
        plt.grid(True)
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        if show: plt.show()
        else   : plt.savefig('%sHist.svg'%name,quality=900)

    """ Figure 04 """
    def fig04(self,useLog=True,show=True,name=''):
        # Parameters
        val = {}

        # Loop
        for code in self.codes:     
            path = os.path.join(self.dir_,self.model,code,'priority.pck')
            df   = self.load(path)
            
            # Compute percentiles
            per = list()
            per.append(np.       min(df['priority']     ))
            per.append(np.percentile(df['priority'],25.0))
            per.append(np.percentile(df['priority'],50.0))
            per.append(np.percentile(df['priority'],75.0))
            #per.append(np.percentile(df['priority'],95.0))
            #per.append(np.percentile(df['priority'],99.0))
            per.append(np.       max(df['priority']     ))

            # Plot
            plt.plot( per )
        if useLog: plt.yscale('log')
        plt.legend(self.codes)
        plt.grid(True)
        if show: plt.show()
        else   : plt.savefig('%sPerc.svg'%name,quality=900)


    """ Table 01 """
    def csv01(self,name):
        val = {}
        # Loop
        for code in self.codes:
            path = os.path.join(self.dir_,self.model,code,'priority.pck')
            df   = self.load(path)

            per = list()
            per.append(np.       min(df['priority']     ))
            per.append(np.percentile(df['priority'],25.0))
            per.append(np.percentile(df['priority'],50.0))
            per.append(np.percentile(df['priority'],75.0))
            per.append(np.percentile(df['priority'],95.0))
            per.append(np.percentile(df['priority'],99.0))
            per.append(np.       max(df['priority']     ))
            val[code] = per

        df = pd.DataFrame(val)
        df.to_csv('per%s.csv'%name,index=False)




non = ["2009081241", "2009130017", "2009161542", "2009191117"]
grd = ["2009061702", "2009041016", "2009020756", "2008311908"]
fll = ["2105092139", "2009230924", "2105100106", "2105121334"]
path = '/media/victor/Documentos/Thesis/Priority/'
"""
obj  = Master('Kim2017',fll,path)
obj.fig03(useLog=True,show=True,name='full')
# obj.fig04(name='Non')

"""
# --------------------------------------------------------------------------------
non = ["2009081241", "2009130017"]      #  0  vs 1.4 [β = 0]
grd = ["2009061702", "2009041016"]      #  0  vs 1.4 [β = 0->1]
fll = ["2105092139", "2009230924"]      #  0  vs 1.4 [β = 1]
fullNo = ["2009230924","2009130017"]    # 1.4 vs 1.4 [β ={1,0}]
fullGr = ["2009230924","2009041016"]    # 1.4 vs 1.4 [β ={1,0->1}]
xmin = 6*10**-5
xmax = 2.0 # 0.5
useLog = True

def load(path):
    with open(path, 'rb') as handle:
        db = pickle.load(handle)
        n_leaf        = db['priority'].n_leaf
        priority      = db['priority']._data[n_leaf-1:]
        n_samples = np.where( priority==0 )[0][0]
        return priority[:n_samples]


codes = non
x = load( os.path.join(path,'Kim2017',codes[0],'priority.pck') )
y = load( os.path.join(path,'Kim2017',codes[1],'priority.pck') )

"""
if useLog:
    x = np.log10(x)
    y = np.log10(y)
    xmin = np.log10(xmin)
    xmax = np.log10(xmax)
t = np.linspace(xmin,xmax,1000)[1:]

plt.figure(figsize=(5,5))
plt.hist2d( x,y, bins=300,vmin=1, vmax=1000, range=[[xmin,xmax],[xmin,xmax]],rasterized=True,norm=LogNorm())#, cmap=plt.cm.Greys)
#plt.colorbar()
#plt.scatter(x,y,rasterized=True,s=0.1,alpha=0.1)
plt.plot(t,t,':r')
#plt.xscale('log')
#plt.yscale('log')
plt.xlim([xmin,xmax])
plt.ylim([xmin,xmax])
#plt.show()
plt.savefig('img.svg',quality=900)
"""

px = list()
py = list()
t = np.linspace(0,100,1000)[1:]
for i in t:
    px.append( np.percentile(x,i) )
    py.append( np.percentile(y,i) )
u = np.linspace(xmin,xmax,1000)[1:]


fig = plt.figure(figsize=(2.5,2.5))
ax = fig.add_subplot(111)
ax.plot(px,py,linewidth=1.5)
ax.plot(u,u,':r',linewidth=1.5)
ax.scatter( np.percentile(x,25) , np.percentile(y,25) ,marker='X',s=20)
ax.scatter( np.percentile(x,50) , np.percentile(y,50) ,marker='X',s=20)
ax.scatter( np.percentile(x,75) , np.percentile(y,75) ,marker='X',s=20)


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([xmin,xmax])
ax.set_ylim([xmin,xmax])
ax.set_aspect('equal')
#plt.show()
plt.grid()
plt.savefig('no.svg',quality=900)


"""
def perVs(x,y,show=True,name='per'):
    xmin = 6*10**-5
    xmax = 2.0 # 0.5
    size = 2.5 # 2.5

    px = list()
    py = list()
    t = np.linspace(0,100,1000)[1:]
    for i in t:
        px.append( np.percentile(x,i) )
        py.append( np.percentile(y,i) )
    u = np.linspace(xmin,xmax,1000)[1:]

    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111)

    ax.plot(px,py,linewidth=1.5)
    ax.plot(u,u,':r',linewidth=1.5)
    ax.scatter( np.percentile(x,25),np.percentile(y,25),marker='o',s=15)
    ax.scatter( np.percentile(x,50),np.percentile(y,50),marker='X',s=15)
    ax.scatter( np.percentile(x,75),np.percentile(y,75),marker='o',s=15)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([xmin,xmax])
    #ax.set_aspect('equal')
    if show: plt.show()
    else   : plt.savefig('%s.svg'%name,quality=900)

# noBias0,noBias1
# grBias0,grBias1
# fuBias0,fuBias1
perVs(noBias0,noBias1,show=False,name='noBias')
perVs(grBias0,grBias1,show=False,name='grBias')
perVs(fuBias0,fuBias1,show=False,name='fuBias')
"""