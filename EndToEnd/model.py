import numpy as np

""" 
Base Model 
*-*-*-*-*-
"""
class BaseModel():
    def __init__(self,config):
        self.config = config    # Config
        self.model  =   None    # Model's net

    def save(self):
        self.model.save_session()

    def load(self):
        self.model.restore_session()
