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
        self.device  = self.init.device
        
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
        checkpath = os.path.join(self.setting.general.savedPath,name,'Model','model'+epoch+'.pth')
        checkpoint = torch.load(checkpath)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Dataset
        self.dataset = CARLA100Dataset(setting,train=False,index='study.csv')

        # Path outputs
        self.path = os.path.join(self.setting.general.savedPath,name)


    """ Transfer host data to device """
    def transfer2device(self,batch):
        inputs    = ['frame','actions','speed','command','mask']
        dev_batch = {}

        for ko in inputs:
            if ko in batch:
                dev_batch[ko] = batch[ko].to(self.device)
        return dev_batch


    """ Hidden state study """
    def hiddenStudy(self):
        # Loader
        imID = self.dataset.generateIDs(True)
        loader = DataLoader(Dataset(self.dataset,imID),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers)
        hiddenControl = list()

        # Model iteration
        self.model.eval()
        print('Hidden state study')
        with torch.no_grad(), tqdm(total=len(loader),leave=False) as pbar:
            for sample in loader:
                # Model
                batch,_ = sample
                dev_batch = self.transfer2device(batch)
                dev_pred = self.model(dev_batch)

                # Extract signals
                host_hc = dev_pred['hidden']['control'].data.cpu().numpy()
                hiddenControl.append(host_hc)

                pbar. update()
                pbar.refresh()
            pbar.close()

        # Save
        print('Save hidden state')
        path = os.path.join(self.path,'hidden.pck')
        with open(path, 'wb') as handle:
            pickle.dump(hiddenControl, handle, protocol=pickle.HIGHEST_PROTOCOL)


    """ Attention study """
    def attentionStudy(self):
        # Loader
        imID = self.dataset.generateIDs(True)
        loader = DataLoader(Dataset(self.dataset,imID),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers)
        spatial     = list()
        categorical = list()

        # Model iteration
        self.model.eval()
        print('Attention study')
        with torch.no_grad(), tqdm(total=len(loader),leave=False) as pbar:
            for sample in loader:
                # Model
                batch,_ = sample
                dev_batch = self.transfer2device(batch)
                dev_pred = self.model(dev_batch)

                # Extract signals
                host_spt = dev_pred['attention']['alpha'].data.cpu().numpy()
                host_cat = dev_pred['attention'][ 'beta'].data.cpu().numpy()
                spatial    .append(host_spt)
                categorical.append(host_cat)

                pbar. update()
                pbar.refresh()
            pbar.close()

        # Save
        print('Save spatial attention')
        path = os.path.join(self.path,'spatial.pck')
        with open(path, 'wb') as handle:
            pickle.dump(spatial, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Save categorical attention')
        path = os.path.join(self.path,'categorical.pck')
        with open(path, 'wb') as handle:
            pickle.dump(categorical, handle, protocol=pickle.HIGHEST_PROTOCOL)


class VisualizingData(object):
    """ Constructor """
    def __init__(self,init,setting,name):
        self.init    =    init
        self.setting = setting
        self.files   = V.FilesForStudy100
        self.device  = self.init.device
