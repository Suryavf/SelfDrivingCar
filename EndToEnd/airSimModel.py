import os
import pandas as pd

from EndToEnd.model import BaseModel
from EndToEnd.network.airnet import AirNet
import Cooking

class AirSimModel(BaseModel):

    def __init__(self,config):
        super(AirSimModel,self).__init__(config)
        self.config = config

        self.model = AirNet(config)

    def precompute(self):
        # Define folders
        folders = ['normal_1', 'normal_2', 'normal_3', 
                   'normal_4', 'normal_5', 'normal_6', 
                   'swerve_1', 'swerve_2', 'swerve_3']

        paths = [os.path.join(self.config.dir_data, f) for f in folders]
        
        """
        dataframes = []
        for folder in paths:
            current = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')
            current['Folder'] = folder
            dataframes.append(current)

        # Define dataset
        dataset = pd.concat(dataframes, axis=0)
        print('Number of data points: {0}'.format(dataset.shape[0]))
        """

        # Cooking data
        Cooking.cook(paths, self.config.dir_cooked, 
                            self.config.train_eval_test_split)


    def build(self):
        self.model.build()



    def load(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def play(self):
        pass

