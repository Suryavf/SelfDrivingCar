import os
import glob

import pickle
from   tqdm import tqdm

import torch
from   torch.utils.data import DataLoader

from openTSNE import TSNE

import StateOfArt.ImitationLearning.ImitationNet as imL
import StateOfArt.        Attention.AttentionNet as attn
import ImitationLearning.VisualAttention.Model   as exper

import ImitationLearning.VisualAttention.Decoder           as D
import ImitationLearning.VisualAttention.Encoder           as E
import ImitationLearning.VisualAttention.network.Attention as A
import ImitationLearning.VisualAttention.network.Control   as C

import common.directory as V
import common.  figures as F
import common.    utils as U
from   common.data import CARLA100Dataset
from   common.data import  GeneralDataset as Dataset


class CookData(object):
    """ Constructor """
    def __init__(self,init,setting,name,epoch):
        self.init    =    init
        self.setting = setting
        self.files   = V.FilesForStudy100
        
        # Modules
        module = {}
        _mod = self.setting.modules
        for k in _mod:
            if   _mod[k] in V.Encoder  : module[  'Encoder'] = eval('E.'+_mod[  'Encoder'])
            elif _mod[k] in V.Decoder  : module[  'Decoder'] = eval('D.'+_mod[  'Decoder'])
            elif _mod[k] in V.Control  : module[  'Control'] = eval('C.'+_mod[  'Control'])
            elif _mod[k] in V.Attention: module['Attention'] = eval('A.'+_mod['Attention'])
            else : raise NameError('ERROR 404: module '+k+' no found')

        # Model
        if   self.setting.model == 'Experimental': self.model = exper. Experimental(module,setting)
        elif self.setting.model == 'ExpBranch'   : self.model = exper.    ExpBranch(module,setting)
        else: print("ERROR: mode no found (" + self.setting.model + ")")

        # Load model checkpoint
        path = os.path.join(self.setting.general.savedPath,name,'Model','model'+epoch+'.pth')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Dataset
        self.dataset = CARLA100Dataset(setting,train=False,index='study.csv')

    """ Hidden state study """
    def hiddenStudy(self,name,epoch):
        # Load 
        imID = self.dataset.generateIDs(True)
        loader = DataLoader(Dataset(self.dataset,imID),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers)

