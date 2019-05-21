class Config(object):
    # Path files
    validPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqVal"
    trainPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqTrain"
    modelPath = "/media/victor/Datos/Tesis/SelfDrivingCar/ImitationLearning/network"
    n_filesPerGroup = 187

    batch_size = 8

    # Net settings
    imageShape   = (88,200,3)
    activation   = 'relu'
    padding      = 'same'
    convDropout  = 0.2
    fullyDropout = 0.5

    # Adam Optimizer
    

