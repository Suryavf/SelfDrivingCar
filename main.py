

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

    parser.add_argument("--steps",type=int,default=50000000,help="Steps for train")
    parser.add_argument("--steps",type=int,default=50000000,help="Steps for train")
