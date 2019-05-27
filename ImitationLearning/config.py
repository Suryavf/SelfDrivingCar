import math

class Config(object):
    # Path files
    validPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqVal"
    trainPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqTrain"
    modelPath = "/media/victor/Datos/Tesis/SelfDrivingCar/ImitationLearning/network"
    graphPath = "/media/victor/Datos/Tesis/SelfDrivingCar/ImitationLearning/network/Graph"
    filesPerGroup = 100#187

    srate           = 10
    epochs          = 30*3600*srate  # 30 horas
    batch_size      = 120
    num_samples     = 1500*200
    epoch_per_save  = 10*60*srate    # 10 minutos 
    steps_per_epoch = math.ceil(num_samples/batch_size)
    
    # Net settings
    imageShape   = (88,200,3)
    activation   = 'relu'
    padding      = 'same'
    convDropout  = 0.2
    fullyDropout = 0.5

    # Learning rate
    learning_rate_initial      = 0.0002
    learning_rate_decay_steps  = 50000
    learning_rate_decay_factor = 0.5

    # Adam Optimizer
    adam_lrate  = 0.0002 
    adam_beta_1 = 0.7 
    adam_beta_2 = 0.85

    # Loss
    lambda_steer = 0.45*0.95
    lambda_gas   = 0.45*0.95
    lambda_brake = 0.05*0.95
    lambda_speed = 0.05

