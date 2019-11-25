import glob
import os
import sys
import random
import math

try:
    sys.path.append('~/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
except IndexError:
    pass

import numpy as np
import  cv2  as cv
import carla

import environment.sensors as S

"""
Carla environment
=================
Ref:
    https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/
    https://carla.readthedocs.io/en/latest/python_api_tutorial/
    https://github.com/carla-simulator/carla/blob/master/Docs/python_api_tutorial.md
"""
class CarlaSim(object):
    """ Constructor """
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
        self.agent  = None
        self.actors = list()

        # Connect to CARLA
        self._connect()

        # Sensors
        self.camera = None

     
    # Connect to the Carla enviroment
    # -------------------------------
    def _connect(self):
        try:
            # Connect to client
            self.client = carla.Client(self.host,self.port)
            self.client.set_timeout(5.0)

            self.world     = self.client.get_world()
            self.blueprint = self. world.get_blueprint_library()

        finally:
            print('Error in connection D:')
    
    # Spawn actors
    # ------------
    def _spawnActors(self):
        # Possible points
        points = self.world.get_map().get_spawn_points()

        # Main actor (random position)
        bp = random.choice(self.blueprint.filter('vehicle.audi.*'))
        self.agent = self.world.spawn_actor(bp, random.choice(points) )
        self.actors.append(self.agent)
        print('Created %s' % self.agent.type_id)

        # Extra vehicles
        for _ in range(self.n_vehicles):
            bp = random.choice(self.blueprint.filter('vehicle'))
            npc = self.world.try_spawn_actor(bp, random.choice(points))
            if npc is not None:
                self.actors.append(npc)
                npc.set_autopilot()
                print('Created %s' % npc.type_id)
        
    """
    Reset enviroment
    ----------------
    """
    def reset(self):
        # Initialize
        self.actors = list()
        self.agent  = None

        # Actors spawn
        self._spawnActors()

        # Sensors spawn
        self.camera = S.CameraRgb(self.world,self.agent)

        # Here's first workaround. If we do not apply any control it takes almost a second for car to start moving
        # after episode restart. That starts counting once we aplly control for a first time.
        # Workarounf here is to apply both throttle and brakes and disengage brakes once we are ready to start an episode.
        # Ref: https://github.com/Sentdex/Carla-RL/blob/master/sources/carla.py
        self.agent.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))


    """
    Get simulation state
    --------------------
    """
    def state(self):
        # RGB camera
        img = self.camera.get()

        # Velocity
        v = self.agent.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        return {'frame': img,
                'speed': kmh}


    """
    Set action on simulation
    ------------------------
    Ref: https://github.com/carla-simulator/carla/blob/master/Docs/python_api_tutorial.md
         https://carla.readthedocs.io/en/latest/python_api_tutorial/
    """
    def setAction(self,steer,gas,brake):
        self.agent.apply_control(carla.VehicleControl(steer=steer, throttle=gas, brake=brake))
    
    
    """
    Destroy actors
    --------------
    """
    def _destroyActor(self,_actor):
        # If it has a callback attached, remove it first
        if hasattr(_actor, 'is_listening') and _actor.is_listening:
            _actor.stop()

        # If it's still alive - desstroy it
        if _actor.is_alive:
            _actor.destroy()
    def destroy(self):
        # Destroy actors
        for carrito in self.actors:
            if carrito is not None:
                self._destroyActor(carrito)

        # Destroy sensors
        self.camera.destroy()
        
