# CARLA Enviroment 

## Necessary Packages

You must choose between

* CARLA :smiley_cat:
* AirSim

## Install CARLA

### Dependences

Install libraries

```shell
pip install pygame numpy future networkx
```

Install UnReal 4.22. Make sure you are registered with Epic Games. This is required to get source code access for Unreal Engine.

```shell
git clone --depth=1 -b 4.22 https://github.com/EpicGames/UnrealEngine.git ~/UnrealEngine_4.22
cd ~/UnrealEngine_4.22
./Setup.sh 
./GenerateProjectFiles.sh 
make
```

### Build CARLA

```shell
git clone https://github.com/carla-simulator/carla
cd ~/carla
./Update.sh
export UE4_ROOT=~/UnrealEngine_4.21
make launch 
make PythonAPI
make package
make help
```

#### Updating 

```shell
git clone https://github.com/carla-simulator/carla
cd ~/carla
./Update.sh
export UE4_ROOT=~/UnrealEngine_4.21
cd ~/carla
make clean
git pull
./Update.sh
make launch
```

#### Download models, and maps 

```shell
git lfs clone https://bitbucket.org/carla-simulator/carla-content Unreal/CarlaUE4/Content/Carla
```


