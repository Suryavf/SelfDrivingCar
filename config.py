import math

class Global(object):
    framePerSecond = 10
    num_workers    =  8
    stepView       = 50 # Print in Train

class Config(object):
    # Path files
    validPath = "./data/h5file/SeqVal"
    trainPath = "./data/h5file/SeqTrain"
    cookdPath = "./data/Cooked"
    modelPath = "./data/Model"
    graphPath = "./data/Model/Graph"
    filesPerGroup = 100#187

    # Model
    model      = 'Basic' # Basic, Multimodal, Codevilla18, Codevilla19
    n_epoch    = 150
    batch_size = 120
    time_demostration = 10*3600

    # Learning rate
    learning_rate_initial      = 0.0002
    learning_rate_decay_steps  = 50000
    learning_rate_decay_factor = 0.5

    # Adam Optimizer
    adam_lrate  = 0.0002 
    adam_beta_1 = 0.7 
    adam_beta_2 = 0.85

    # Loss
    lambda_steer = 0.45
    lambda_gas   = 0.45
    lambda_brake = 0.05
    lambda_action = 0.95
    lambda_speed  = 0.05
    
