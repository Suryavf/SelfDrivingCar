import sys
import argparse
import config

from ReinforcementLearning.dqnAgent import DQNAgent

class Main():

    def __init__(self,init,setting):

        self.setting = setting
        self.init    = init


        self.model = None
        """
        # Basic End-To-End
        if s.model == "endToEnd":
            from EndToEnd.config import Config
            self.model = EndToEndModel(Config())
        # CIL
        elif s.model == "cil":
            self.model = CILmodel(s.city)

        # Basic DQN
        elif s.model == "dqn":
            from ReinforcementLearning.config import Config
            self.model = DQNAgent(Config())
        else:
            print("Invalid model type")
        """

    def train(self):
        self.model.train()

    def test (self):
        self.model.test()

    def play (self):
        self.model.play()


if __name__ == "__main__":

    # Parser define
    parser = argparse.ArgumentParser(description="SelfDriving")

    # Path
    parser.add_argument("--trainpath" ,type=str,help="Data for train")
    parser.add_argument("--validpath" ,type=str,help="Data for validation")
    parser.add_argument("--savedpath" ,type=str,help="Data for saved data")
    parser.add_argument("--n_epoch"   ,type=str,help="Number of epoch for train")
    parser.add_argument("--batch_size",type=str,help="Batch size for train")

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