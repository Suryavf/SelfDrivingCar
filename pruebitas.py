from os      import listdir
from os.path import isfile, join    
import h5py

path = "/home/zoser/Descargas/CORL2017ImitationLearningData/AgentHuman/SeqVal"    
fileList = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]

for p in fileList:
    # Data
    print("File:",p)
    file = h5py.File(p, 'r')
    #print("Read check")

    #print("RGB:",    file['rgb'].value.shape)
    #print("RGB check")
    #print("Targets:",file['targets'].value.shape)
    #print("Targets check")
    #print("\n")