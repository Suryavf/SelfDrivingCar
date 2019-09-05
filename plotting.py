import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
basic       = "/media/victor/Datos/Saved/Basic1908112006/Model/model.csv"
multimodal  = "/media/victor/Datos/Saved/Multimodal1908121930/Model/model.csv"
codevilla18 = "/media/victor/Datos/Saved/Codevilla181908201218/Model/model.csv"
codevilla19 = "/media/victor/Datos/Saved/Codevilla19/1908270053/Model/model.csv"
#"/media/victor/Datos/Saved/Codevilla191908221756/Model/model.csv"
kim2017     = "/media/victor/Datos/Saved/Kim20171908151827/Model/model.csv"


basic       = pd.read_csv(      basic,index_col=0)
multimodal  = pd.read_csv( multimodal,index_col=0)
codevilla18 = pd.read_csv(codevilla18,index_col=0)
codevilla19 = pd.read_csv(codevilla19,index_col=0)
kim17       = pd.read_csv(    kim2017,index_col=0)

n_epoch = [ len(basic),len(multimodal),
            len(codevilla18),len(codevilla19),len(kim17),100 ]

epochs = range(1, np.max(n_epoch)+1 )


def plotLines(category,title,y_min,y_max):
	plt.figure(figsize=(10,4.5))
	plt.plot(range(1, len(      basic)+1 ),basic      [category])
	plt.plot(range(1, len( multimodal)+1 ),multimodal [category])
	plt.plot(range(1, len(codevilla18)+1 ),codevilla18[category])
	plt.plot(range(1, len(codevilla19)+1 ),codevilla19[category])
	plt.plot(range(1, len(      kim17)+1 ),kim17      [category])
	plt.legend(['Basic','Multimodal',
		        'Codevilla 2018','Codevilla 2019',
		        'Kim 2017'])#, bbox_to_anchor=(1.0,1.0))#(0.7,0.55))
	plt.title(title)
	plt.ylim(y_min,y_max)
	plt.xlim(1,100)
	plt.savefig("images/"+category+".svg")#show()

plotLines('LossTrain',     'Train loss',0.045,0.160)
plotLines('LossValid','Validation loss',0.080,0.155)
plotLines('Steer'    ,          'Steer',0.020,0.038)
plotLines('Gas'      ,            'Gas',0.060,0.120)
plotLines('Brake'    ,          'Brake',0.088,0.165)
"""

Adam   = "/media/victor/Datos/Saved/Kim2017/1908151827/Model/model.csv"
RAdam  = "/media/victor/Datos/Saved/Kim2017/1909011125/Model/model.csv"
Ranger = "/media/victor/Datos/Saved/Kim2017/1909031232/Model/model.csv"

Adam   = pd.read_csv(  Adam,index_col=0)
RAdam  = pd.read_csv( RAdam,index_col=0)
Ranger = pd.read_csv(Ranger,index_col=0)

n_epoch = [ len(Adam),len(RAdam),len(Ranger),100 ]

epochs = range(1, np.max(n_epoch)+1 )

def plotLines(category,title,y_min,y_max):
	plt.figure(figsize=(10,4.5))
	plt.plot(range(1, len(  Adam)+1 ),Adam  [category])
	plt.plot(range(1, len( RAdam)+1 ),RAdam [category])
	plt.plot(range(1, len(Ranger)+1 ),Ranger[category])
	plt.legend(['Adam','RAdam','Ranger'])
	plt.title(title)
	plt.ylim(y_min,y_max)
	plt.xlim(1,100)
	plt.show()#savefig("images/"+category+".svg")#show()

plotLines('LossTrain',     'Train loss',0.045,0.160)
plotLines('LossValid','Validation loss',0.080,0.155)
