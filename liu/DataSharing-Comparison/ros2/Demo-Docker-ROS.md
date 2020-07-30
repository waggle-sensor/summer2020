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
#### Save mount changes permanently
```
sudo vi /etc/fstab 
```
```
#add the following commands into /etc/fstab: 

/dev/nvme0n1p1  /var/lib/docker   ext4 defaults 0  2
```

### Pull Docker images from Docker hub and start the Docker container

#### For ROS1 (if there is no USB camera connected, remove "--device=/dev/video0:/dev/video1"):
```
$ sudo docker pull liangkailiu/plugin-tensorflow-ros:v1.9.7
$ sudo xhost +si:localuser:root
$ sudo docker run --runtime nvidia  --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 liangkailiu/plugin-tensorflow-ros:v1.9.7
```
#### For ROS2 (if there is no USB camera connected, remove "--device=/dev/video0:/dev/video1"):
```
$ sudo docker pull liangkailiu/plugin-tensorflow-ros:v2.0.7
$ sudo xhost +si:localuser:root
$ sudo docker run --runtime nvidia  --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 liangkailiu/plugin-tensorflow-ros:v2.0.7
```

#### Launch several containers with the same Docker images
Add "--name XX"
```
$ sudo docker run --runtime nvidia -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 --name MyContainer2 liangkailiu/plugin-tensorflow-ros:v2.0.7
$ sudo docker run --runtime nvidia -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 --name MyContainer3 liangkailiu/plugin-tensorflow-ros:v2.0.7
$ sudo docker run --runtime nvidia -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 --name MyContainer4 liangkailiu/plugin-tensorflow-ros:v2.0.7   
```

### Set up object detection application runnning inside ROS1
1. Launch USB camera images publisher:
```
$ roslaunch usb_cam usb_cam-test.launch
```
2. Launch YOLO-based object detection application:
```
$ roslaunch darknet_ros yolo_v3.launch
```
3. Inside other containers, check the bandwidth, frequency, etc. using rostopic:
```
$ rostopic bw /images   # bandwidth
$ rostopic hz /images   # frequency
$ rostopic echo /images # print full msg
```

### Set up object detection application runnning inside ROS2
1. Launch USB camera images publisher
```
$ ros2 run image_tools cam2image 1
```
2. Launch YOLO-based object detection application
```
$ ros2 launch darknet_ros darknet_ros.launch.py
```
3. Inside other containers, check the bandwidth, frequency, etc. using rostopic:
```
$ ros2 topic bw /images
$ ros2 topic hz /images
$ ros2 topic echo /images
```
