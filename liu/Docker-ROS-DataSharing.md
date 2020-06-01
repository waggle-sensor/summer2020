## Setup ROS communications between Docker containers:

Run Docker containers with “--network host”, containers created on the same machine will be assigned the same IP address as the host.

![image](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/Docker-ROS-demo.png)

### [Run ROS across multiple machines](http://wiki.ros.org/ROS/Tutorials/MultipleMachines)
Note: two machines should be able to ping each other

### ROS communications between containers on the same machine:

Launch docker Container 1:
```
$ sudo xhost +si:localuser:root
$ sudo docker run --runtime nvidia --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 liangkailiu/plugin-tensorflow-ros:v5
```
Container 1 uses the host network so it has an IP address 172.17.0.1.
```
root@nvidia-desktop:/# ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255
        inet6 fe80::42:cbff:fe1e:42eb  prefixlen 64  scopeid 0x20<link>
        ether 02:42:cb:1e:42:eb  txqueuelen 0  (Ethernet)
        RX packets 179  bytes 10946 (10.9 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 289  bytes 503959 (503.9 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```
Set ROS master URI on Container 1:
```
root@nvidia-desktop:~# export ROS_MASTER_URI=http://172.17.0.1:11311
```
Launch docker Container 2:
```
$ sudo xhost +si:localuser:root
$ sudo docker run --runtime nvidia -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video0:/dev/video1 liangkailiu/plugin-tensorflow-ros:v5
```
Container 2 is launched without the host network so it has an IP assigned by Docker, which is 172.17.0.2. Container 1 and 2 are in the same local network.
```
root@f0577834616f:/# ifconfig
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.17.0.2  netmask 255.255.0.0  broadcast 172.17.255.255
        ether 02:42:ac:11:00:02  txqueuelen 0  (Ethernet)
        RX packets 169  bytes 322755 (322.7 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 116  bytes 9113 (9.1 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

Set ROS master URI on Container 2:
```
root@nvidia-desktop:~# export ROS_MASTER_URI=http://172.17.0.1:11311
```

### ROS communications between containers on the different machines:
A router is used to connect the two devices together into the same LAN. The setup is the same as the first one.

Container 1:
```
root@nvidia-desktop:/# export ROS_MASTER_URI=http://192.168.0.234:11311
root@nvidia-desktop:/# roslaunch darknet_ros darknet_ros.launch 
... logging to /root/.ros/log/5c596132-a129-11ea-9f5e-00044be58e94/roslaunch-nvidia-desktop-138.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.

started roslaunch server http://nvidia-desktop:43619/
...
Loading weights from /root/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/yolov2-tiny.weights...Done!
Waiting for image.
```

Container 2:
```
root@nvidia-desktop:/# export ROS_MASTER_URI=http://192.168.0.234:11311
root@nvidia-desktop:/# rostopic list
/darknet_ros/bounding_boxes
/darknet_ros/check_for_objects/cancel
/darknet_ros/check_for_objects/feedback
/darknet_ros/check_for_objects/goal
/darknet_ros/check_for_objects/result
/darknet_ros/check_for_objects/status
/darknet_ros/detection_image
/darknet_ros/found_object
/rosout
/rosout_agg
/usb_cam/image_raw
```

## [Running ROS across multiple REMOTE machines](http://wiki.ros.org/ROS/Tutorials/MultipleRemoteMachines)
 - Cloud-based: facilitate remote ROS networks by transferring the contents of the ROS messages through the cloud. This would require a two-way data conversion, or rosmsg-webformat-rosmsg bridge for each rosmsg being transferred through the cloud.
 - PortForwarding (PF) 
