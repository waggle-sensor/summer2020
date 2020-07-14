## Install Docker Container with ROS1 / ROS2

### Mount SSD to /var/lib/docker, which stores the docker images 

#### Mount ssdhome and ssddocker to temp mount points
1. Mount ssd to temporary mount points, **/mnt/ssdhome** and **/mnt/ssddocker** 
```
sudo mkdir /mnt/ssddocker  
sudo mount /dev/nvme0n1p1 /mnt/ssddocker  
```
2. Synchronize files in /home and /var/lib/docker with /mnt/ssdhome and /mnt/ssddocker
* if there is no "/var/lib/docker" folder, create it first. or skip the following one command.
```
sudo mkdir -p /var/lib/docker
```
```
sudo rsync -aXS /var/lib/docker/.  /mnt/ssddocker/.
```
#### Change mount points
```
sudo mv /var/lib/docker  /var/lib/docker-old

sudo mkdir -p /var/lib/docker

sudo umount /dev/nvme0n1p1 

sudo mount /dev/nvme0n1p1 /var/lib/docker
```
## Save mount changes permanently
```
sudo vi /etc/fstab 
```
```
#add the following commands into /etc/fstab: 

/dev/nvme0n1p1  /var/lib/docker   ext4 defaults 0  2
```

### Pull Docker images from Docker hub and start the Docker container

#### For ROS1:
```
$ sudo docker pull liangkailiu/plugin-tensorflow-ros:v1.9.7
$ sudo xhost +si:localuser:root
# if there is no USB camera connected, remove "--device=/dev/video0:/dev/video1"
$ sudo docker run --runtime nvidia  --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 liangkailiu/plugin-tensorflow-ros:v1.9.7
```
#### For ROS2:
```
$ sudo docker pull liangkailiu/plugin-tensorflow-ros:v2.0.7
$ sudo xhost +si:localuser:root
$ sudo docker run --runtime nvidia  --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 liangkailiu/plugin-tensorflow-ros:v2.0.7
```

### Set up applications runnning inside ROS1
