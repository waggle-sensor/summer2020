# Daily Notes - Liangkai Liu

## Week 1 (May 4 to May 8)

### May 4 to May 7, 2020
#### Work Done:

**Prototype Setup:** set up a prototype with a Leopard HD camera and a Velodyne VLP16 LiDAR. 
* system configurations
	* Ubuntu 18.04
	* CUDA 10.0
  * TensorFlow 1.14.0
  * OpenCV 3.3.1
  * ROS 1.0 Melodic
  * ...

**Demos on the image/LiDAR point cloud processing**
* four demos: 
  * object detection on USB camera
  * LiDAR point cloud capturing and viewing
  * LiDAR point cloud segmentation
  * calibration of LiDAR and camera

**Report Link:** https://docs.google.com/document/d/14qHXLcrpSZkLaF3EGXZMHhy75IStIEFUl-iIESRkvUw/edit?usp=sharing

#### To Do List:
* solve the sychronization issue of message from Velodyne LiDAR
* get the calibration result of LiDAR point cloud and camera image
* try more state-of-the-art works on LiDAR/Camera data processing to understand the system requirements

### May 8, 2020
#### Work Done: 
Solve the sychronization issue of message from Velodyne LiDAR. It turns out to be the problem of the driver. Right now the message -/velodyne_points- has the time read from the system clock. **Time sychronization** can be an important issue when multiple sensors need to be used together for some applications. Two things to consider for time sychronization. One is the sychronization granunarity (second/millisecond/microsecond levels), which is mainly determined by the application. The other is how to get "true" time, potential solution is NTP-based or GPS/GNSS-based. 

**object detection using LiDAR point cloud**

**PCDet**: rank #1 on KITTI 3D object detection (http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
depends on SpConv1.0 (https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634), which needs cmake3.13.2+ (ref: https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)

**system requirements to enable Waggle for multi-sensor processing**
 * time sychronization
 * concurrent access to the sensor data (multi-process access the same sensor at the same time)
 * frequency issue: different sensor has different FPS, for example camera has 30 or 60 FPS, LiDAR has 10 FPS. The time to capture a frame/point cloud data from the sensor also vary within a small range. How to manage these issues?
 * to be added .......

#### To Do List:
* calibration result of LiDAR point cloud and camera image
* set up application on transforming point cloud to panorama image
* set up application on using point cloud or the panorama from LiDAR for object detection

## Week 2 (May 11 to May 15)

### May 11, 2020
#### Work Done: 

**object detection using LiDAR point cloud**

Set up dependencies including SpConv1.0 and cmake for PCDet. Issue comes out when install other requirements including numba, tensorboardX, easydict, pyyaml, scikit-image, and tqdm. Use conda install numpy. Issues remain when install others.Issues with the Jetson TX2 board. On another NVIDIA GPU clsuter machine, successfully set up the PCDet.

**system dependency:** an important issue for the deployment of multi-sensor applications. For deep learning-based applications, CUDA, TensorFlow, Torch, cmake, gcc/g++, and other libraries' version need to configure in advance.

#### To Do List:
* Solve the issue on Jetson TX2 and get the object detection using point cloud
* experiments with some results to show the time synchronization, concurrency issues with ROS

### May 12, 2020
#### Work Done: 

**attend the SAGE workshop:** Understand the vision of the SAGE project and its deliverable plan. The presentations have covered questions like how to prepare dataset with labels for the training of DNN models, SAGE programming model with scheudling APIs, edge code repository, the data path from edge to cloud, training data collection, and SAGE data object storage.

#### To Do List:
* Learn related slides and try to understand the objective as well as specific tasks of system support for multi-sensor processing at the edge 
* Set up apps for experiments for using ROS in multi-sensor processing

### May 13, 2020
#### Work Done:
work on experiments of several process (ROS node) subscribe to the same image/point cloud data. Proflile the memory footprint, CPU usage, GPU usage, and power consumption.

**object detection using LiDAR point cloud:** set up point cloud-based object detection demo (PCDet) on a GPU cluster. The training data includes image, calibration data, point cloud, and label.
![image](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/PCDet.jpg)

#### To Do List:
* Finish the experiments in ROS and get a analysis report
* Get a design/plan of the work on optimization


### May 14, 2020
#### Work Done:
Discuss to get the plan for the next two weeks. There will be two specific things: compare the performace in terms of latency, accuracy time, etc of Docker with ROS under Real-Time Kernel and generic kernel; data sharing of images/messages between dockers. After that, I have built a Docker image with ROS based on the image containing machine learning tools.

#### To Do List:
* Implement Real-Time Kernel on Jetson board
* Do experiments using generic kernel with applications running on Docker with ROS

### May 15, 2020
#### Work Done:
 * Set up Real-time kernel patch for Ubuntu 18.04 on Nvidia Jetson TX2. (Ref link: https://stackoverflow.com/questions/51669724/install-rt-linux-patch-for-ubuntu)
 * Set up object detection demo using image in the Docker image
 * Experiments to see the performance of obejct detection demo on generic kernel

#### To Do List:
* Set up generic kernel on Nvidia AGX Xavier board
* Explore the measure tools for the profiling of the application

## Week 3 (May 18 to May 22)

### May 18, 2020
#### Work Done:
 * Set up the docker container with ROS and object detection demo, the set up notes can be find at https://docs.google.com/document/d/1cW1deiGM4HMRzKKOMZVZCw9Ns_-ZDOfSTMaQvUUyjaA/edit?usp=sharing
 * Flash Nvidia AGX Xavier with Ubuntu 18.04 and RT-kernel

#### To Do List:
* Set up generic kernel on Nvidia AGX Xavier board
* Get applications running within the Docker

### May 19, 2020
#### Work Done:
 * Face a no space issue after buinding the Docker image on Jetson TX2 board. On another Intel-based device, make the docker image but the architecture is x86-64, which cannot run on Jetson AGX Xavier (arm64). Get a new Nvidia AGX Xavier and launch the docker container and commit the changes to a new docker image named plugin-tensorflow-ros. There is another issue come out. which shows that "CUDA driver is not sufficient with run environment".

#### To Do List:
* Solve the CUDA drivier version issue
* Get applications running within the Docker

### May 20, 2020
#### Work Done:
 * Try to install all ROS-related packdges on base docker image from Nvidia (https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow). The CUDA driver issue is solved. Another issue realted to OpenCV 4.x comes out, which is because the OpenCV 4.x requires C++ 11+ support. The demo can works with CUDA 10.0 and OpenCV 3.3.1 on Jetson TX2.

#### To Do List:
* Get applications running within the Docker
* Try install generic kernel patch into Nvidia Jetson AGX board

### May 21, 2020
#### Work Done:
 * Try to solve the OpenCV 4.x dependency issue
 * Implement the new docker image from Nvidia on Jetson AGX and built ROS into it
 * Try to install generic kernel on the Nvidia Jetson AGX Xavier

OpenCV4 issue with ROS:
```
nvidia@nvidia-desktop:~$ sudo docker run -it plugin-tensorflow-ros /bin/bash
[sudo] password for nvidia: 
root@e2e7604628c5:/# roslaunch darknet_ros 
darknet_ros.launch      object_detection.test   
darknet_ros_gdb.launch  yolo_v3.launch          
root@e2e7604628c5:/# roslaunch darknet_ros 
darknet_ros.launch      object_detection.test   
darknet_ros_gdb.launch  yolo_v3.launch          
root@e2e7604628c5:/# roslaunch darknet_ros darknet_ros.launch 
... logging to /root/.ros/log/5ab37520-9c47-11ea-a3ab-0242ac110002/roslaunch-e2e7604628c5-211.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.

started roslaunch server http://e2e7604628c5:35317/

SUMMARY
========

PARAMETERS
 * /darknet_ros/actions/camera_reading/name: /darknet_ros/chec...
 * /darknet_ros/config_path: /root/catkin_ws/s...
 * /darknet_ros/image_view/enable_console_output: True
 * /darknet_ros/image_view/enable_opencv: True
 * /darknet_ros/image_view/wait_key_delay: 1
 * /darknet_ros/publishers/bounding_boxes/latch: False
 * /darknet_ros/publishers/bounding_boxes/queue_size: 1
 * /darknet_ros/publishers/bounding_boxes/topic: /darknet_ros/boun...
 * /darknet_ros/publishers/detection_image/latch: True
 * /darknet_ros/publishers/detection_image/queue_size: 1
 * /darknet_ros/publishers/detection_image/topic: /darknet_ros/dete...
 * /darknet_ros/publishers/object_detector/latch: False
 * /darknet_ros/publishers/object_detector/queue_size: 1
 * /darknet_ros/publishers/object_detector/topic: /darknet_ros/foun...
 * /darknet_ros/subscribers/camera_reading/queue_size: 1
 * /darknet_ros/subscribers/camera_reading/topic: /camera/rgb/image...
 * /darknet_ros/weights_path: /root/catkin_ws/s...
 * /darknet_ros/yolo_model/config_file/name: yolov2-tiny.cfg
 * /darknet_ros/yolo_model/detection_classes/names: ['person', 'bicyc...
 * /darknet_ros/yolo_model/threshold/value: 0.3
 * /darknet_ros/yolo_model/weight_file/name: yolov2-tiny.weights
 * /rosdistro: melodic
 * /rosversion: 1.14.5

NODES
  /
    darknet_ros (darknet_ros/darknet_ros)

auto-starting new master
process[master]: started with pid [221]
ROS_MASTER_URI=http://localhost:11311

setting /run_id to 5ab37520-9c47-11ea-a3ab-0242ac110002
process[rosout-1]: started with pid [232]
started core service [/rosout]
process[darknet_ros-2]: started with pid [238]
[ INFO] [1590164093.347287816]: [YoloObjectDetector] Node started.
[ INFO] [1590164093.363454414]: [YoloObjectDetector] Xserver is not running.
[ INFO] [1590164093.375057429]: [YoloObjectDetector] init().
YOLO V3
layer     filters    size              input                output
    0 CUDA Error: CUDA driver version is insufficient for CUDA runtime version
CUDA Error: CUDA driver version is insufficient for CUDA runtime version: Resource temporarily unavailable
[darknet_ros-2] process has died [pid 238, exit code 255, cmd /root/catkin_ws/devel/lib/darknet_ros/darknet_ros camera/rgb/image_raw:=/camera/rgb/image_raw __name:=darknet_ros __log:=/root/.ros/log/5ab37520-9c47-11ea-a3ab-0242ac110002/darknet_ros-2.log].
log file: /root/.ros/log/5ab37520-9c47-11ea-a3ab-0242ac110002/darknet_ros-2*.log
^C[rosout-1] killing on exit
[master] killing on exit
shutting down processing monitor...
... shutting down processing monitor complete
```
Nvidia Jetson docker link issue:
```
nvidia@nvidia-desktop:/usr/local/cuda-10.2/targets/aarch64-linux/lib$ sudo nvidia-container-cli -k -d /dev/tty info

-- WARNING, the following logs are for debugging purposes only --

I0522 15:58:24.937181 20852 nvc.c:281] initializing library context (version=1.1.1, build=e5d6156aba457559979597c8e3d22c5d8d0622db)
I0522 15:58:24.937470 20852 nvc.c:255] using root /
I0522 15:58:24.937503 20852 nvc.c:256] using ldcache /etc/ld.so.cache
I0522 15:58:24.937540 20852 nvc.c:257] using unprivileged user 65534:65534
W0522 15:58:24.938437 20852 nvc.c:171] failed to detect NVIDIA devices
I0522 15:58:24.939162 20853 nvc.c:191] loading kernel module nvidia
E0522 15:58:24.940468 20853 nvc.c:193] could not load kernel module nvidia
I0522 15:58:24.940512 20853 nvc.c:203] loading kernel module nvidia_uvm
E0522 15:58:24.941286 20853 nvc.c:205] could not load kernel module nvidia_uvm
I0522 15:58:24.941325 20853 nvc.c:211] loading kernel module nvidia_modeset
E0522 15:58:24.942163 20853 nvc.c:213] could not load kernel module nvidia_modeset
I0522 15:58:24.943107 20854 driver.c:101] starting driver service
E0522 15:58:24.943938 20854 driver.c:161] could not start driver service: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory
I0522 15:58:24.944365 20852 driver.c:196] driver service terminated successfully
nvidia-container-cli: initialization error: driver error: failed to process request
```

#### To Do List:
* Try to solve the OpenCV4 issue and Nvidia Docker issue
* Try to get generic kernel patch
