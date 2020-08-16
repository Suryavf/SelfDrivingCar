import glob
import os

import json
import cv2 as cv
import numpy as np

import h5py

path = "/media/victor/Datos/Descargas/CVPR2019-CARLA100_14"
out  = "/media/victor/Datos/Carla100"
folders = glob.glob(os.path.join(path,'episode_*'))
folders.sort()

# Descartar   episode_01174   episode_02576   episode_02594   episode_03473   episode_03474   episode_03485
# folders = folders[13:]

for f in folders:
    # Read
    imgs = glob.glob(os.path.join(f,'CentralRGB_*'))
    
    name = os.path.basename(f)
    print('Create '+name)

    rgb      = list()
    command  = list()
    actions  = list()
    velocity = list()

    n_img = len(imgs)
    #print(n_img)
    k = 0
    while k<n_img:
        #for k in range(n_img): #img,pathjson in zip(imgs,msn):
        # Paths
        num = '0'*(5-len(str(k)))+str(k)
        img      = os.path.join(f,  'CentralRGB_'+num+'.png' )
        pathjson = os.path.join(f,'measurements_'+num+'.json')

        if os.path.isfile(pathjson):
            # Add frontal camera
            img = cv.imread(img)
            if img.shape[0] != 88:
                print('img.shape[0] =',img.shape[0],'   k =',k,'\n')
                k = n_img 
            else: 
                rgb.append(img)


                with open(pathjson) as jsonfile:
                    info = json.load(jsonfile)
                    
                    # Actions
                    actions.append( np.array([ info['steer'],info['throttle'],info['brake'] ]) )

                    # Command
                    command.append( info['directions'] )

                    # Velocity
                    if 'forwardSpeed' in info['playerMeasurements']:
                        velocity.append( info['playerMeasurements']['forwardSpeed'] )
                    else:
                        velocity.append( 0 )
        else:
            print('pathjson isnt file k=',k)
            k = n_img 
        k = k + 1

    # To array
    rgb      = np.array(   rgb  )
    actions  = np.array( actions)
    command  = np.array( command)
    velocity = np.array(velocity)

    pathmeta = os.path.join(f,'metadata.json')
    if os.path.isfile(pathmeta):
        with open( pathmeta ) as jsonfile:
            meta = json.load(jsonfile)
            n_pedestrian = meta['number_of_pedestrian']
            n_vehicles   = meta['number_of_vehicles'  ]
            weather      = meta['weather'             ]
    else:
        n_pedestrian = -1
        n_vehicles   = -1
        weather      = -1

    with h5py.File( os.path.join(out,name+'.hdf5'),"w") as f:
        dset = f.create_dataset(     "rgb", data=rgb     )
        dset = f.create_dataset( "actions", data=actions )
        dset = f.create_dataset( "command", data=command )
        dset = f.create_dataset("velocity", data=velocity)
        dset = f.create_dataset("number_of_pedestrian", data=n_pedestrian)
        dset = f.create_dataset("number_of_vehicles"  , data=n_vehicles  )
        dset = f.create_dataset("weather"             , data=weather     )
        
