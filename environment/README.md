# SelfDrivingCar

## Necessary Packages

You must choose between

* CARLA :smiley_cat:
* AirSim

## Install CARLA

### Dependences

```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update

sudo apt-get install build-essential clang-6.0 lld-6.0 g++-7 cmake ninja-build python python-pip python-dev python3-dev python3-pip 
libpng16-dev 
libtiff5-dev libjpeg-dev tzdata sed curl wget unzip autoconf libtool

pip2 install --user setuptools
pip3 install --user setuptools
```

Change your default clang version to compile Unreal Engine and the CARLA dependencies

```shell
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-6.0/bin/clang++ 102
sudo update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-6.0/bin/clang 102
```

### UnReal 4.21

Make sure you are registered with Epic Games. This is required to get source code access for Unreal Engine.

```shell
git clone --depth=1 -b 4.21 https://github.com/EpicGames/UnrealEngine.git ~/UnrealEngine_4.21
cd ~/UnrealEngine_4.21
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

