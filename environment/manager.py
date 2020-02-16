from carla.client           import CarlaClient,make_carla_client
from carla.settings         import CarlaSettings
from carla.sensor           import Camera
from carla.carla_server_pb2 import Control
from carla.planner.planner  import  Planner
from carla.tcp              import TCPConnectionError
from carla.client           import VehicleControl
import environment.config  as carla_config

import signal
import subprocess
import random
import time
import os
import numpy as np
from PIL  import Image
from enum import Enum

class action_space(object):
    def __init__(self, dim, high, low, seed):
        self.shape = (dim,)
        self. high = np.array(high)
        self.  low = np.array(low )
        self. seed = seed
        assert(dim == len(high) == len(low))
        np.random.seed(self.seed)

    def sample(self):
        return np.random.uniform(self.low, self.high)

class observation_space(object):
    def __init__(self, dim, high=None, low=None, seed=None):
        self.shape = (dim,)
        self. high = high
        self.  low = low
        self. seed = seed

class FinishState(Enum):
    TIME_OUT = 0
    COLLISION_VEHICLE = 1
    COLLISION_PEDESTRIAN = 2 
    COLLISION_OTHER = 3
    OFFROAD = 4


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
    def __init__(self, log_dir,image_agent,city="/Game/Maps/Town01"):
        self.log_dir = log_dir
        self.carla_server_settings =None
        self.server = None
        self.server_pid = -99999
        self.map = city

        self.client = None  #carla client 
        self.  host = 'localhost'
        self.  port = 2000
        
        self.Image_agent = image_agent

        #steer,throttle,brake
        #self.action_space = action_space(3, (1.0, 1.0,1.0), (-1.0,0,0), SEED)
        #featured image,speed,steer,other lane ,offroad,
        #collision with pedestrians,vehicles,other
        #self.observation_space = observation_space(512 + 7)
        self.max_episode = 1000000
        self.time_out_step = 10000
        self.max_speed = 35
        self.speed_up_steps = 20 
        self.current_episode = 0
        self.weather = -1
        self.current_step = 0
        self.current_position = None
        self.total_reward = 0
        self.planner = None
        self.carla_setting = None
        self.number_of_vehicles = None
        self.control = None
        self.nospeed_times =0
        self.reward = 0
        self.observation = None
        self.done =False
        self.LoadConfig()
        self.Setup()


    """ Load Configuration """
    def LoadConfig(self):
        self.   vehicle_pair = carla_config.NumberOfVehicles
        self.pedestrian_pair = carla_config.NumberOfPedestrians
        self.   weather_set  = carla_config.set_of_weathers
        #[straight,one_curve,navigation,navigation]
        if self.map=="/Game/Maps/Town01":
            self.poses = carla_config.poses_town01()
        elif self.map=="/Game/Maps/Town02":
            self.poses = carla_config.poses_town02()
        else:
            print("Unsupported Map Name")


    """ Setup client """
    def Setup(self):
        self.client = CarlaClient(self.host, self.port, timeout=99999999) #carla  client 
        self.client.connect(connection_attempts=100)


    def reset(self):
        self.nospeed_times =0 
        pose_type = random.choice(self.poses)
        #pose_type =  self.poses[0]
        self.current_position = random.choice(pose_type)  #start and  end  index
        #self.current_position = (53,67)
        self.number_of_vehicles    = 0  # random.randint( self.vehicle_pair[0],self.vehicle_pair[1])
        self.number_of_pedestrians = 0  # random.randint( self.vehicle_pair[0],self.vehicle_pair[1])
        self.weather = random.choice(self.weather_set)

        settings = carla_config.make_carla_settings()
        settings.set(   NumberOfVehicles = self.number_of_vehicles,
                     NumberOfPedestrians = self.number_of_pedestrians,
                               WeatherId = self.weather    )
        self.carla_setting = settings
        self.scene = self.client.load_settings(settings)
        self.client.start_episode(self.current_position[0]) #set the start position
        
        self.target_transform = self.scene.player_start_spots[self.current_position[1]]
        self.planner = Planner(self.scene.map_name)

        # Skip the  car fall to sence frame
        for i in range(self.speed_up_steps): 
            self.control = VehicleControl()
            self.control.steer = 0
            self.control.throttle = 0.025*i
            self.control.brake = 0
            self.control.hand_brake = False
            self.control.reverse = False
            time.sleep(0.05)
            send_success = self.send_control(self.control)
            if not send_success:
                return None
            self.client.send_control(self.control)

        measurements, sensor_data = self.client.read_data() #measurements,sensor 
        directions =self.getDirections(measurements,self.target_transform,self.planner)
        if directions is None or measurements is None:
            return None
        state,_,_=self.get_state(measurements,sensor_data,directions)
        return state 


    """ Send Control """
    def send_control(self,control):
        send_success = False
        try:
            self.client.send_control(control)
            send_success = True
        except Exception:
            print("Send Control error")
        return send_success


    """ Get Directions: function to get the high level commands and the waypoints 
        The waypoints correspond to the local planning, the near path the car has to follow.
    """
    def getDirections(self,measurements, target_transform, planner):
        # Get the current position from the measurements
        current_point = measurements.player_measurements.transform
        try:
            directions = planner.get_next_command(
                            (   current_point.location   .x,    current_point.location   .y,                0.22           ),
                            (   current_point.orientation.x,    current_point.orientation.y,    current_point.orientation.z),
                            (target_transform.location   .x, target_transform.location   .y,                0.22           ),
                            (target_transform.orientation.x, target_transform.orientation.y, target_transform.orientation.z))
        except Exception:
            print("Route plan error ")
            directions = None
        return directions


    """  Get state
        comute new state,reward,and is done
    """
    def get_state(self,measurements,sensor_data,directions):
        # Frame
        img_feature = self.Image_agent.compute_feature(sensor_data)  #shape = (512,)
        
        # Measurements
        speed                  = measurements.player_measurements.forward_speed # m/s
        intersection_offroad   = measurements.player_measurements.intersection_offroad
        intersection_otherlane = measurements.player_measurements.intersection_otherlane
        collision_vehicles     = measurements.player_measurements.collision_vehicles
        collision_pedestrians  = measurements.player_measurements.collision_pedestrians
        collision_other        = measurements.player_measurements.collision_other

        self.reward = 0 
        done = False 


    def _getImage(self):
        pass
    def _getImage(self):
        pass
    def _getReward(self):
        pass

