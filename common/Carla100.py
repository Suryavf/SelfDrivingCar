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

# Skip 
blacklist = ['episode_01936','episode_02594','episode_03476','episode_03590']

n_folders = len(folders)
for i,f in enumerate(folders):
    # Read
    name = os.path.basename(f)
    print('Create %s \t\t %i/%i'%(name,i+1,n_folders)) 
    
    # Outfile path
    outfile = os.path.join(out,name+'.h5')
    if os.path.exists(outfile):
        print ('Skip by repetition\n')
        continue

    # Blacklist
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
            if img is None:
                print('It is None D:')
                rgb = []
                k = n_img # END
            elif img.shape[0] != 88:
                print('No shape match (%i,%i)\t[itr %i]'%(img.shape[0],img.shape[1],k))
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
            print('File not found (iter %i)'%k)
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
        print('Few frames (%i)\n'%len(rgb)) 
        
