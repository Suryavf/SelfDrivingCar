import math

class Global(object):
    framePerSecond =  10
    num_workers    =   8
    stepView       =  10 # Print in Train
    max_steering   = 1.2
    max_speed      =  90

class Config(object):
    # Path files
    validPath = "./data/h5file/SeqVal/"
    trainPath = "./data/h5file/SeqTrain/"
    cookdPath = "/media/victor/Datos/Cooked"
    savedPath = "/media/victor/Datos/Saved"
    filesPerGroup = 100#187

    # Model
    model      = 'Multimodal' # Basic, Multimodal, Codevilla18, Codevilla19
    n_epoch    =  80
    batch_size = 120
    time_demostration = 72000 # 20 h

    # Learning rate
    learning_rate_initial      = 0.0001
    learning_rate_decay_steps  = 10
    learning_rate_decay_factor = 0.5

    # Adam Optimizer
    adam_lrate  = 0.0001
    adam_beta_1 = 0.7#0.9  #0.7 
    adam_beta_2 = 0.85#0.999#0.85

    # Loss
    lambda_steer = 0.45
    lambda_gas   = 0.45
    lambda_brake = 0.05
    lambda_action = 0.95
    lambda_speed  = 0.05
    
