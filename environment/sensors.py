import numpy as np
import  cv2  as cv

import carla
from   carla import ColorConverter as cc

"""
    RGB camera
    -------------------
    Ref: https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/driving_benchmarks/corl2017/corl_2017.py
         https://carla.readthedocs.io/en/latest/cameras_and_sensors/
         https://github.com/carla-simulator/carla/blob/master/Docs/python_api_tutorial.md
         https://carla.readthedocs.io/en/latest/python_api/#carlatransform-class
"""
class CameraRgb(object):
    display = None

    def __init__(self, world, actor):
        # Display camera
        self.display = None

        # Find the blueprint of the sensor.
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')

        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', f'{800}')
        blueprint.set_attribute('image_size_y', f'{600}')
        blueprint.set_attribute('fov', '100')

        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', '0.1')

        # Provide the position of the sensor relative to the vehicle.
        transform = carla.Transform(carla.Location(x=2.0, z=1.4),carla.Rotation(-15.0, 0, 0))

        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.camera = world.spawn_actor(blueprint, transform, attach_to=actor)

        self.camera.listen(lambda image: self.__on_sensor_event(image))

    #@staticmethod
    def __on_sensor_event(self,image):
        # Carla image to array (numpy)
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :,  : 3]
        
        # Crop
        array = array[90:485,:]
        
        self.display = cv.resize(array,(200,88),interpolation=cv.INTER_CUBIC)

    def get(self):
        return self.display

    def destroy(self):
        # If it has a callback attached, remove it first
        if hasattr(self.camera, 'is_listening') and self.camera.is_listening:
            self.camera.stop()

        # If it's still alive - desstroy it
        if self.camera.is_alive:
            self.camera.destroy()
            
