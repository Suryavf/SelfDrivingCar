import os
import argparse
import config

from common.utils import str2bool
#from ReinforcementLearning.dqnAgent import DQNAgent
from ImitationLearning.ImitationModel import ImitationModel

_imitationLearningList = ['Basic','Multimodal','Codevilla18','Codevilla19','Kim2017']

class Main():
    """ Constructor """
    def __init__(self,init,setting):
        self.model   = None

        # Define seed
        init.set_seed()
        
        if setting.model in _imitationLearningList:
            self.model = ImitationModel(init,setting)
        else:
            raise NameError('ERROR 404: Model no found')
        self.model.build()
        
    def load(self,path):
        self.model.load(path)
    def to_continue(self,name):
        self.model.to_continue(name)
    def train(self):
        self.model.execute()
    def plot(self,name):
        self.model.plot(name)
    def play(self):
        pass
    

if __name__ == "__main__":

    # Parser define
    parser = argparse.ArgumentParser(description="SelfDriving")

    # Path
    parser.add_argument("--trainpath"  ,type=str,help="Data for train")
    parser.add_argument("--validpath"  ,type=str,help="Data for validation")
    parser.add_argument("--savedpath"  ,type=str,help="Path for saved data")
    parser.add_argument("--modelpath"  ,type=str,help="Model file path")
    parser.add_argument("--settingpath",type=str,help="Setting folder path")
    parser.add_argument("--n_epoch"    ,type=int,help="Number of epoch for train")
    parser.add_argument("--batch_size" ,type=int,help="Batch size for train")
    parser.add_argument("--model"      ,type=str,help="End-to-End model: Basic, Multimodal, Codevilla18, Codevilla19, Kim2017")

    parser.add_argument("--optimizer",type=str     ,help="Optimizer method: Adam, RAdam, Ranger")
    parser.add_argument("--scheduler",type=str2bool,help="Use scheduler (boolean)")

    parser.add_argument("--name",type=str,help="Code model.")
    parser.add_argument("--mode",default="train",type=str,help="Select execution mode: train,continue,play,plot")
    args = parser.parse_args()

    # Setting  
    init    = config.   Init()
    setting = config.Setting()

    # Load setting
    if args.settingpath is not None:
        init   .load( os.path.join(args.settingpath,   "init.json") )
        setting.load( os.path.join(args.settingpath,"setting.json") )
    
    # Model
    if args.model is not None: setting.model = args.model
    
    # Path
    if args.trainpath  is not None: setting.general.trainPath = args.trainpath
    if args.validpath  is not None: setting.general.validPath = args.validpath
    if args.savedpath  is not None: setting.general.savedPath = args.savedpath
    
    # Train
    if args.n_epoch    is not None: setting.train.n_epoch    = args.n_epoch
    if args.batch_size is not None: setting.train.batch_size = args.batch_size
    
    if args.optimizer is not None: setting.train.optimizer.type      = args.optimizer
    if args.scheduler is not None: setting.train.scheduler.available = args.scheduler

    # Print settings
    setting.print()

    # Loaded modes
    if args.mode in ['continue','plot']:
        init.is_loadedModel = True

    # Main program
    main = Main(init,setting)
    
    # Load model
    if args.modelpath is not None:
        main.load(args.modelpath)

    # Execute mode
    if   args.mode == "train":
        main.train()
    elif args.mode == "play":
        main.play()
    elif args.mode == "continue":
        if args.name is not None: 
            
            main.to_continue(args.name)
        else:
            NameError('Undefined model. Please define with --name"')
    elif args.mode == "plot":
        if args.name is not None: 
            main.plot(args.name)
        else:
            NameError('Undefined model. Please define with --name"')
    else:
        print("Valid execution modes: train,play")
        
