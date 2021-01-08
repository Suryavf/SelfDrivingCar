import glob
import os

import json
import cv2 as cv
import numpy as np

import h5py

path = "/home/suryavf/raw"
out  = "/home/suryavf/CARLA100"
folders = glob.glob(os.path.join(path,'episode_*'))
folders.sort()

# Descartar   episode_01174   episode_02576   episode_02594   episode_03473   episode_03474   episode_03485
# folders = folders[13:]

for f in folders:
    # Read
    name = os.path.basename(f)
    print('Create '+name)

    # Outfile path
    outfile = os.path.join(out,name+'.hdf5')
    if os.path.exists(outfile):
        print ('Skip by repetition\n')
        continue

    # Blacklist
    blacklist = ['episode_01936']
    if name in blacklist:
        print('Skip by blacklist\n')
        continue

    # Omite bad episodes
    if os.path.exists(os.path.join(f,'bad_episode')):
        print ('BAD EPISODE!\n')
        continue
    
    rgb      = list()
    command  = list()
    actions  = list()
    velocity = list()

    imgs = glob.glob(os.path.join(f,'CentralRGB_*'))
    n_img = len(imgs)
    k = 0
    while k<n_img:
        # Paths
        num = '0'*(5-len(str(k)))+str(k)
        img      = os.path.join(f,  'CentralRGB_'+num+'.png' )
        pathjson = os.path.join(f,'measurements_'+num+'.json')

        if os.path.isfile(pathjson) & os.path.isfile(img):
            # Add frontal camera
            img = cv.imread(img)
            if img.shape[0] != 88:
                print('img.shape[0] =',img.shape[0],'   k =',k,'\n')
                k = n_img # END
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
            print('pathjson is not file k=',k)
            k = n_img # END
        k = k + 1
    
    if len(rgb)>120:
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

        with h5py.File(outfile,"w") as f:
            dset = f.create_dataset(     "rgb", data=rgb     )
            dset = f.create_dataset( "actions", data=actions )
            dset = f.create_dataset( "command", data=command )
            dset = f.create_dataset("velocity", data=velocity)
            dset = f.create_dataset("number_of_pedestrian", data=n_pedestrian)
            dset = f.create_dataset("number_of_vehicles"  , data=n_vehicles  )
            dset = f.create_dataset("weather"             , data=weather     )
    else:
        print('Few frames\n')        
