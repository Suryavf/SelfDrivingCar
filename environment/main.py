# ready to run example: PythonClient/car/hello_car.py
import airsim
import time

# connect to the AirSim simulator
# https://buildmedia.readthedocs.org/media/pdf/airsim-fork/docs/airsim-fork.pdf
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

while True:
    # get state of the car
    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # set the controls for car
    car_controls.throttle = 1
    car_controls.steering = 1
    client.setCarControls(car_controls)

    # let car drive a bit
    time.sleep(1)

    # get camera images from the car
    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis   ),
                                     airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)]) 
    print('Retrieved images: %d', len(responses))

    # do something with images
    for response in responses:
        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            airsim.write_file('py1.png', response.image_data_uint8)