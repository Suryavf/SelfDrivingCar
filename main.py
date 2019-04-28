

import sys
import argparse


class Main():

    def __init__(self,modelType,conf):
        self.model = None

    def train(self):
        self.model.train()

    def test (self):
        self.model.test()

    def play (self):
        self.model.play()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SelfDriving")

    parser.add_argument("-steps",type=int,default=50000000,help="Steps for train")
    parser.add_argument("-data" ,type=str,default="./data",help="Data for train/test")
    parser.add_argument("-nets" ,type=str,default="./nets",help="Path of model train")
    arg,remaining = parser.parse_known_args()
