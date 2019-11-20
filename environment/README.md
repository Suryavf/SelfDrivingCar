# CARLA Enviroment 

## Necessary Packages

You must choose between

* CARLA :smiley_cat:
* AirSim

## Install CARLA

### Dependences

Install libraries

```shell
pip install pygame numpy
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


## Install AIRsim

### Dependences

```shell
sudo -H pip3 install msgpack-rpc-python
sudo apt-get install libboost-all-dev
sudo apt install libgconf-2-4
```


### UnReal 4.18

Make sure you are registered with Epic Games. This is required to get source code access for Unreal Engine.

```shell
git clone -b 4.18 https://github.com/EpicGames/UnrealEngine.git
cd UnrealEngine
./Setup.sh
./GenerateProjectFiles.sh
make
```


### Unity
Check last version in
<a href="https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/page-2">link</a>.


```shell
wget http://beta.unity3d.com/download/292b93d75a2c/UnitySetup-2019.1.0f2
chmod +x UnitySetup-2019.1.0f2
./UnitySetup-2019.1.0f2
```

Another instructions in 
<a href="https://github.com/Microsoft/AirSim/blob/master/Unity/README.md">link</a>.


### AirSim

Clone AirSim and build it

```shell
git clone https://github.com/Microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

