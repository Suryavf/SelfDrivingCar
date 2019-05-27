class Config(object):
    # Path files
    validPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqVal"
    trainPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqTrain"
    modelPath = "/media/victor/Datos/Tesis/SelfDrivingCar/ImitationLearning/network"
    graphPath = "/media/victor/Datos/Tesis/SelfDrivingCar/ImitationLearning/network/Graph"
    filesPerGroup = 100#187

    batch_size = 6
    epochs     = 120

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
    
    # Loss
    lambda_steer = 0.45*0.95
    lambda_gas   = 0.45*0.95
    lambda_brake = 0.05*0.95
    lambda_speed = 0.05