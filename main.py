import sys
import argparse
from utils import Setting

from EndToEnd.endToEndModel import EndToEndModel
from ReinforcementLearning.dqnAgent import DQNAgent

class Main():

    def __init__(self,s):
        self.model = None

        # Basic End-To-End
        if s.model == "endToEnd":
            from EndToEnd.config import Config
            self.model = EndToEndModel(Config())

        # Basic DQN
        elif s.model == "dqn":
            from ReinforcementLearning.config import Config
            self.model = DQNAgent(Config())
        else:
            print("Invalid model type")


    def train(self):
        self.model.train()

    def test (self):
        self.model.test()

    def play (self):
        self.model.play()


if __name__ == "__main__":

    # Parser define
    parser = argparse.ArgumentParser(description="SelfDriving")

    parser.add_argument("--data" ,type=str,default="./data"   ,help="Data for train/test")
    parser.add_argument("--nets" ,type=str,default="./nets"   ,help="Path of model train")
    parser.add_argument("--host" ,type=str,default="localhost",help="IP of the host server (default: localhost)")
    parser.add_argument("--port" ,type=int,default=2000       ,help="TCP port to listen to (default: 2000)")
    parser.add_argument("--city" ,type=str,default="Town01"   ,help="The town that is going to be used on benchmark") #Town01/Town02
    parser.add_argument("--mode" ,type=str,default="train"    ,help="Select execution mode: train,test,play")
    parser.add_argument("--model",type=str,default="model"    ,help="Agents model")
    args = parser.parse_args()


    # Setting  
    s = Setting()
    s.datapath = args.data
    s. netpath = args.nets
    s.    host = args.host
    s.    port = args.port
    s.    city = args.city

    # Main program
    main = Main(s)

    if   args.mode == "train":
        main.train()
    elif args.mode == "test":
        main.test()
    elif args.mode == "play":
        main.play()
    else:
        print("Valid execution modes: train,test,play")