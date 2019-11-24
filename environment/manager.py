import glob
import os
import sys
import random

try:
    sys.path.append('~/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
except IndexError:
    pass

import numpy as np
import  cv2  as cv
import carla
 
"""
Carla environment
=================
Ref:
    https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/
    https://carla.readthedocs.io/en/latest/python_api_tutorial/
    https://github.com/carla-simulator/carla/blob/master/Docs/python_api_tutorial.md
"""
class CarlaSim(object):
    def __init__(self,host='localhost',port=2000):
        # Parameters
        self.n_vehicles = 10
        self.host       = host
        self.port       = port

        # Objects
        self.client    = None
        self.world     = None
        self.blueprint = None
        
        # Actors
        self.actor    = None
        self.vehicles = list()

        # Sensors
        self.camera = None

        self.controls = None
        self.isPrinted = False
 

    """
    Connect to the Carla enviroment
    -------------------------------
    """
    def connect(self):
        try:
            # Connect to client
            self.client = carla.Client(self.host,self.port)
            self.client.set_timeout(5.0)

            self.world     = self.client.get_world()
            self.blueprint = self.world.get_blueprint_library()

        finally:
            print('Error in connection D:')


    """
    Spawn actors
    ------------
    """
    def spawnActor(self):
        # Possible points
        points = self.world.get_map().get_spawn_points()

        # Main actor (random position)
        bp = random.choice(self.blueprint.filter('vehicle.audi.*'))
        self.actor = self.world.spawn_actor(bp, random.choice(points) )
        print('Created %s' % self.actor.type_id)

        # Extra vehicles
        for _ in range(self.n_vehicles):
            bp = random.choice(self.blueprint.filter('vehicle'))
            npc = self.world.try_spawn_actor(bp, random.choice(points))
            if npc is not None:
                self.vehicles.append(npc)
                npc.set_autopilot()
                print('Created %s' % npc.type_id)
        

    """
    Set camara settings
    -------------------
    Ref: https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/driving_benchmarks/corl2017/corl_2017.py
         https://carla.readthedocs.io/en/latest/cameras_and_sensors/
         https://github.com/carla-simulator/carla/blob/master/Docs/python_api_tutorial.md
    """
    def setCamera(self):
        # Find the blueprint of the sensor.
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', '800')
        blueprint.set_attribute('image_size_y', '600')
        blueprint.set_attribute('fov', '100')

        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', '0.1')

        # Provide the position of the sensor relative to the vehicle.
        transform = carla.Transform(carla.Location(x=2.0, z=1.4),carla.Rotation(-15.0, 0, 0))

        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.camera = self.world.spawn_actor(blueprint, transform, attach_to=self.actor)

    
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
    def _preprocessingImage(self,image):
        # Carla image to array (numpy)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :,  : 3]
        array = array[:, :, ::-1]

        # Crop
        array = array[90:485,:]
        
        return cv.resize(array,(200,88),interpolation=cv.INTER_CUBIC)

    def getCameraImage(self):
        self.camera.listen(lambda image: self._preprocessingImage(image))
        
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
    Ref: https://github.com/carla-simulator/carla/blob/master/Docs/python_api_tutorial.md
         https://carla.readthedocs.io/en/latest/python_api_tutorial/
    """
    def setAction(self,steer,gas,brake):
        self.actor.apply_control(carla.VehicleControl(steer=steer, throttle=gas, brake=brake))
