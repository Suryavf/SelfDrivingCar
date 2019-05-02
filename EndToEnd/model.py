import numpy as np

""" 
Base Model 
*-*-*-*-*-
"""
class BaseModel():
    def __init__(self,config):
        self.config = config    # Config
        self.net    =   None    # Model's net

    def save(self):
        self.net.save_session()

    def load(self):
        self.net.restore_session()
