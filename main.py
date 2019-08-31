import sys
import argparse
import config

from common.utils import str2bool
from ReinforcementLearning.dqnAgent import DQNAgent
from ImitationLearning.ImitationModel import ImitationModel

_imitationLearningList = ['Basic','Multimodal','Codevilla18','Codevilla19','Kim2017']

class Main():

    def __init__(self,init,setting):

        self.setting = setting
        self.init    = init

        self.model = None

        if setting.model in _imitationLearningList:
            self.model = ImitationModel(init,setting)
        else:
            raise NameError('ERROR 404: Model no found')

        self.model.build()

    def train(self):
        self.model.execute()
    """
    def test (self):
        self.model.test()

    def play (self):
        self.model.play()
    """

if __name__ == "__main__":

    # Parser define
    parser = argparse.ArgumentParser(description="SelfDriving")

    # Path
    parser.add_argument("--trainpath" ,type=str,help="Data for train")
    parser.add_argument("--validpath" ,type=str,help="Data for validation")
    parser.add_argument("--savedpath" ,type=str,help="Data for saved data")
    parser.add_argument("--n_epoch"   ,type=int,help="Number of epoch for train")
    parser.add_argument("--batch_size",type=int,help="Batch size for train")

    parser.add_argument("--optimizer",type=str     ,help="Optimizer method: Adam, RAdam, Ranger")
    parser.add_argument("--scheduler",type=str2bool,help="Use scheduler (boolean)")

    parser.add_argument("--mode" ,type=str,help="Select execution mode: train,test,play")
    parser.add_argument("--model",type=str,help="Agents model")
    args = parser.parse_args()

    # Setting  
    init    = config.   Init()
    setting = config.Setting()

    # Path
    if args.trainpath  is not None: setting.general.trainPath = args.trainpath
    if args.validpath  is not None: setting.general.validPath = args.validpath
    if args.savedpath  is not None: setting.general.savedPath = args.savedpath

    # Train
    if args.n_epoch    is not None: setting.train.n_epoch    = args.n_epoch
    if args.batch_size is not None: setting.train.batch_size = args.batch_size
    
    if args.optimizer is not None: setting.train.optimizer.optimizer = args.optimizer
    if args.scheduler is not None: setting.train.scheduler.available = args.scheduler

    # Main program
    main = Main(init,setting)
    """
    if   args.mode == "train":
        main.train()
    elif args.mode == "test":
        main.test()
    elif args.mode == "play":
        main.play()
    else:
        print("Valid execution modes: train,test,play")

    """