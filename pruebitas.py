from os      import listdir
from os.path import isfile, join    
import h5py

path = "/media/victor/Datos/CORL2017ImitationLearningData/AgentHuman/SeqTrain"    
fileList = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]

#for p in fileList:
    # Data
    #print("File:",p)
    #file = h5py.File(p, 'r')
    
    #print("RGB:",    file['rgb'].value.shape)
    #print("RGB check")
    #if file['targets'].value.shape[0] >200:
    #    print("Targets:",file['targets'].value.shape[0])
    #print("Targets check")
    #print("\n")
    
from common.preprocessing import fileH5py

file = fileH5py(fileList[2])

oneHot = file.commandOneHot()

