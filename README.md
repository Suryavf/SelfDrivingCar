# SelfDrivingCar

## Necessary Packages

* Unity (o UnReal) :smiley_cat:
* AirSim

### Install AIRsim

#### Dependences

```shell
sudo -H pip3 install msgpack-rpc-python
sudo apt-get install libboost-all-dev
sudo apt install libgconf-2-4
```


#### UnReal

Make sure you are registered with Epic Games. This is required to get source code access for Unreal Engine.

```shell
git clone -b 4.18 https://github.com/EpicGames/UnrealEngine.git
cd UnrealEngine
./Setup.sh
./GenerateProjectFiles.sh
make
```


#### Unity
Check last version in
<a href="https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/page-2">link</a>.


```shell
wget http://beta.unity3d.com/download/292b93d75a2c/UnitySetup-2019.1.0f2
chmod +x UnitySetup-2019.1.0f2
./UnitySetup-2019.1.0f2
```

More instructions in 
<a href="https://github.com/Microsoft/AirSim/blob/master/Unity/README.md">link</a>.


#### AirSim

Clone AirSim and build it

```shell
git clone https://github.com/Microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

