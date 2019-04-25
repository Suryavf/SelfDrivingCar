import airsim
import time
import numpy as np

"""
AirSim environment
==================
Ref:
    https://buildmedia.readthedocs.org/media/pdf/airsim-fork/docs/airsim-fork.pdf
"""
class AirSimCar:
    def __init__(self):
        self.controls =  None
        self.client   =  None
        
        self.isPrinted = False


    """
    Connect to the AirSim simulator
    -------------------------------
    """
    def connect(self):
        # Connection
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
        # Get control info
        self.controls = airsim.CarControls() 

    
    """
    Get simulation state
    --------------------
    """
    def getState(self):
        state = self.client.getCarState()
        if self.isPrinted:
            print("Speed %d, Gear %d" % (state.speed, state.gear))
        return state


    """
    Get camera images from simulation
    ---------------------------------
    """
    def getCameraImage(self):
        scene = self.client.simGetImage("0",airsim.ImageType.Scene)

        # get numpy array
        img1d   = np.fromstring(scene.image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 4
        img_rgba = img1d.reshape(scene.height, scene.width, 4)
        # original image is fliped vertically
        img_rgba = np.flipud(img_rgba)
        
        return img_rgba


    """
    Set action on simulation
    ------------------------
    """
    def setAction(self,control):
        self.client.setCarControls(control)