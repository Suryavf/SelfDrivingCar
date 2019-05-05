from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

class Config(object):

    train_steps            = 50000000
    batch_size             = 64
    history_len            = 4
    frame_skip             = 4
    epsilon_start          = 1.0
    epsilon_end            = 0.05
    epsilon_decay_episodes = 1000000
    epsilon_decay          = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
    max_steps              = 10000
    
    train_freq  = 8
    update_freq = 10000
    train_start = 20000
    dir_save    = "saved_session/"
    restore     = False
    
    random_start = 10
    test_step    = 5000
    network_type = "dqn"

    # Learning
    gamma                 = 0.99
    learning_rate_minimum = 0.00025
    lr_method             = "rmsprop"
    learning_rate         = 0.00025
    lr_decay              = 0.97
    keep_prob             = 0.8

    min_history = 4
    states_to_update = 4

    """
    The meaning of each letter might help understand:

     - N: number of images in the batch
     - H: height of the image
     - W: width of the image
     - C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)

    """
    if get_available_gpus():
        cnn_format = "NCHW"
    else:
        cnn_format = "NHWC"

