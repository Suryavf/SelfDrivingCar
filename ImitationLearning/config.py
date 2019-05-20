class Config(object):
    # Path files
    valPath   = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqVal"
    trainPath = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqTrain"
    n_filesPerGroup = 187

    # Net settings
    imageShape   = (88,200,3)
    activation   = 'relu'
    padding      = 'same'
    convDropout  = 0.2
    fullyDropout = 0.5

    # Compile
    

