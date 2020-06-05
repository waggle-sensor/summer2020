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

### May 22, 2020
#### Work Done:
* Flash several edge devices for experiments. Below is the table containing the linux and library version.

| Machine ID |         Device        | Ubuntu | CUDA | OpenCV | TensorFlow |
|:----------:|:---------------------:|:------:|:----:|:------:|:----------:|
|      1     |       Jetson TX2      |  18.04 | 10.0 |  3.2.0 |   1.15.0   |
|      2     |      AGX Xavier1      |  18.04 | 10.2 |  4.1.1 |      -     |
|      3     |      AGX Xavier2      |  18.04 | 10.2 |  4.1.1 |      -     |
|      4     | Intel Fog (CPU-based) |  18.04 |   -  |  4.2.0 |   1.14.0   |

 * Try the Yolov3 object detection demo (ROS Darknet: https://github.com/leggedrobotics/darknet_ros) on machine 1 and 2. 
 * Try Tensroflow object detection demo (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html) on all four machines

#### To Do List:
* Investigate system level tools for profiling the application running on real-time/generic kernel
* Get generic kernel patch into Nvidia Jetson AGX board

## Week 4 (May 26 to May 29)

### May 26, 2020
#### Work Done:
* Solve several issues of running object detection demo in Docker+ROS, including cv_bridge with python3 issue, OpenCV4 with ROS issue, and cv_bridge with OpenCV4 issue. Details can be found at [ROS-Docker-Setup.pdf](https://github.com/waggle-sensor/summer2020/blob/master/liu/ROS-Docker%20Setup.pdf).

* Try RT kernel patch and generic kernel patch on AGX Xavier device

#### To Do List:
* Set up system level tools like perf, htop, and strace for profiling the application running on real-time/generic kernel

### May 27, 2020
#### Work Done:
* Solve dependency issues of running Darknet in Docker ROS and build the complete Docker image, currently the application is ready and the Docker image has been pushed onto Docker Hub. Some library versions that work well with ROS 1.0 Melodic are: CUDA 10.2, OpenCV 3.4.0, Python 3.6, and TensorFlow 1.15.2. The command to pull the Docker image (18GB):
```
docker pull liangkailiu/plugin-tensorflow-ros:v5
```
* Discuss with Raj and try RT kernel patch kernel patch on AGX Xavier device. The challenge is to find a propriate linux kernel with low latency or real-time patch but can build on arm64 architecture.

#### To Do List:
* Try and solve the issues while building premptive kernel patch for Nvidia AGX Xavier 
* Set up system level tools like perf, htop, and strace for profiling the application running on real-time/generic kernel

### May 28, 2020
#### Work Done:
* Try to build RT-kernel using corss-compile on an x86 machine (Ref: https://medium.com/@r7vme/real-time-kernel-for-nvidia-agx-xavier-b660e107a211) but gets some errors when impleemtning on Jetson AGX. The issues may be caused on the JetPack versions. 
* Finish the data sharing between Docker containers using ROS. There are two types of data sharing: between containers on the same device; between containers on two Jetson device. Setup details can be found at: https://docs.google.com/document/d/14_v1tB89duOA5YHmGE8Q24396NS8CckfU07jGvw7fPk/edit?usp=sharing. 

#### To Do List:
* Solve the issue of building RT kernel on Nvidia Jetson AGX
* Start the comparison experiments on Docker+ROS with generic/RT kernels

### May 29, 2020
#### Work Done:
* Prepare demo for the data sharing of ROS messages between Docker containers. Setup details can be found at: https://docs.google.com/document/d/14_v1tB89duOA5YHmGE8Q24396NS8CckfU07jGvw7fPk/edit?usp=sharing. The feature of data sharing between Docker containers is finished.
* Clean up notes for setting up ROS in Docker and achieving data sharing in Docker with ROS. The notes for Docker ROS setup can be found at [here](https://github.com/waggle-sensor/summer2020/blob/master/liu/Docker-ROS-Setup.md) and the notes for data sharing in Docker with ROS can be found at [here](https://github.com/waggle-sensor/summer2020/blob/master/liu/Docker-ROS-DataSharing.md).

#### To Do List:
* Solve the issue of building RT kernel on Nvidia Jetson AGX
* Experiments of the latency, system resource utilization, energy, etc of the data sharing demo

## Week 5 (June 1 to June 5)

### June 1, 2020
#### Work Done:
* Sprint 3 meeting to determine the plan of work for next two weeks
* Prepare slides for the presentation to LBNL about the data sharing using Docker and ROS 1.0. Check out the slides [here](https://github.com/waggle-sensor/summer2020/blob/master/liu/Meeting%20slides%20-%20LBNL%20-%2006:01:20.pdf). 

#### To Do List:
* Solve the issue of building RT kernel on Nvidia Jetson AGX
* Experiments of the latency, system resource utilization, energy, etc of the data sharing demo

### June 2, 2020
#### Work Done:
* Set up experiemnts for the Docker ROS demo. Docker stat and top command will be used to measure the system resource. Latency with the breakdown will be recorded based on the ROS message header.
* Try setting up of RT kernel based on [Nvidia L4T](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fkernel_custom.html).

#### To Do List:
* Build RT kernel on Nvidia Jetson AGX
* Porformance profiling of the data sharing demo

### June 3, 2020
#### Work Done:
* Set up RT kernel on Jetson AGX Xavier with L4T 32.2.1. The setting up notes can be found at: https://github.com/waggle-sensor/summer2020/blob/master/liu/doc/L4T-RTKernel-Setup.md.
* Flash tools include CUDA, OpenCV, cuDNN, etc into AGX board with Jetpack 4.3.
* Try RT kernel test based on https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/rt-tests.

```
sudo apt-get install build-essential libnuma-dev
git clone git://git.kernel.org/pub/scm/utils/rt-tests/rt-tests.git
cd rt-tests
git checkout stable/v1.0
make all
make install
```
The output of the generic kernel test:
```
nvidia@nvidia-xavier-rt:~/projects/rt-tests$ sudo ./cyclictest --mlockall --smp --priority=80 --interval=200 --distance=0
# /dev/cpu_dma_latency set to 0us
policy: fifo: loadavg: 1.04 0.74 0.61 3/1212 8147           

T: 0 ( 8137) P:80 I:200 C:  73055 Min:      7 Act:   13 Avg:   15 Max:     151
T: 1 ( 8138) P:80 I:200 C:  73040 Min:      7 Act:   18 Avg:   15 Max:     125
T: 2 ( 8139) P:80 I:200 C:  73004 Min:      7 Act:   39 Avg:   14 Max:    1076
T: 3 ( 8140) P:80 I:200 C:  72989 Min:      6 Act:   48 Avg:   14 Max:     141
```
Config RT kernel to work:
```
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
```
The output of the RT kernel test:
```
nvidia@nvidia-xavier-rt:~/projects/rt-tests$ uname -a
Linux nvidia-xavier-rt 4.9.140-rt93-tegra #1 SMP PREEMPT RT Wed Jun 3 16:58:57 EDT 2020 aarch64 aarch64 aarch64 GNU/Linux
nvidia@nvidia-xavier-rt:~/projects/rt-tests$ sudo ./cyclictest --mlockall --smp --priority=80 --interval=200 --distance=0
# /dev/cpu_dma_latency set to 0us
policy: fifo: loadavg: 0.34 0.61 0.60 1/1238 8573           

T: 0 ( 8565) P:80 I:200 C:  18697 Min:      4 Act:    7 Avg:    8 Max:     116
T: 1 ( 8566) P:80 I:200 C:  18669 Min:      4 Act:    9 Avg:    7 Max:      49
T: 2 ( 8567) P:80 I:200 C:  18673 Min:      3 Act:    7 Avg:    7 Max:     109
T: 3 ( 8568) P:80 I:200 C:  18660 Min:      4 Act:    6 Avg:    7 Max:      45
T: 4 ( 8569) P:80 I:200 C:  18648 Min:      4 Act:    5 Avg:    8 Max:      83
T: 5 ( 8570) P:80 I:200 C:  18635 Min:      4 Act:    7 Avg:    7 Max:      46
T: 6 ( 8571) P:80 I:200 C:  18623 Min:      4 Act:    6 Avg:    7 Max:      43
T: 7 ( 8572) P:80 I:200 C:  18611 Min:      4 Act:    6 Avg:    7 Max:      50
```

#### To Do List:
* Set up RT-kernel and generic kernel comparison experiments
* Experiments of profiling the Docker with ROS 1.0 

### June 4, 2020
#### Work Done:
* Try JetPack 4.4 and JetPack 4.3 with the RT kernel based AGX board and run plugin-tensorflow-ros Docker image inside. Get issue on the CUDA version not sufficient error.

The kernel log of nvidia-container-cli shows that the RT-kernel can support up to CUDA 10.0:
```
sudo nvidia-container-cli -k -d /dev/tty info

-- WARNING, the following logs are for debugging purposes only --

I0605 05:22:37.213574 8689 nvc.c:281] initializing library context (version=0.9.0+beta1, build=77c1cbc2f6595c59beda3699ebb9d49a0a8af426)
I0605 05:22:37.213708 8689 nvc.c:255] using root /
I0605 05:22:37.213726 8689 nvc.c:256] using ldcache /etc/ld.so.cache
I0605 05:22:37.213749 8689 nvc.c:257] using unprivileged user 65534:65534
I0605 05:22:37.214068 8690 driver.c:134] starting driver service
I0605 05:22:37.227025 8689 nvc_info.c:585] requesting driver information with ''
I0605 05:22:37.227960 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-tls.so.32.3.1
I0605 05:22:37.228107 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.32.3.1
I0605 05:22:37.228210 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-glsi.so.32.3.1
I0605 05:22:37.228296 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-glcore.so.32.3.1
I0605 05:22:37.228416 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-fatbinaryloader.so.32.3.1
I0605 05:22:37.228509 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-eglcore.so.32.3.1
I0605 05:22:37.229494 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1
I0605 05:22:37.229981 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libGLX_nvidia.so.0
I0605 05:22:37.230072 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/libGLESv2_nvidia.so.2
I0605 05:22:37.230146 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/libGLESv1_CM_nvidia.so.1
I0605 05:22:37.230228 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/libEGL_nvidia.so.0
W0605 05:22:37.230330 8689 nvc_info.c:305] missing library libnvidia-ml.so
W0605 05:22:37.230343 8689 nvc_info.c:305] missing library libnvidia-cfg.so
W0605 05:22:37.230353 8689 nvc_info.c:305] missing library libnvidia-opencl.so
W0605 05:22:37.230362 8689 nvc_info.c:305] missing library libnvidia-compiler.so
W0605 05:22:37.230370 8689 nvc_info.c:305] missing library libvdpau_nvidia.so
W0605 05:22:37.230391 8689 nvc_info.c:305] missing library libnvidia-encode.so
W0605 05:22:37.230400 8689 nvc_info.c:305] missing library libnvcuvid.so
W0605 05:22:37.230408 8689 nvc_info.c:305] missing library libnvidia-fbc.so
W0605 05:22:37.230417 8689 nvc_info.c:305] missing library libnvidia-ifr.so
W0605 05:22:37.230426 8689 nvc_info.c:309] missing compat32 library libnvidia-ml.so
W0605 05:22:37.230436 8689 nvc_info.c:309] missing compat32 library libnvidia-cfg.so
W0605 05:22:37.230453 8689 nvc_info.c:309] missing compat32 library libcuda.so
W0605 05:22:37.230462 8689 nvc_info.c:309] missing compat32 library libnvidia-opencl.so
W0605 05:22:37.230471 8689 nvc_info.c:309] missing compat32 library libnvidia-ptxjitcompiler.so
W0605 05:22:37.230501 8689 nvc_info.c:309] missing compat32 library libnvidia-fatbinaryloader.so
W0605 05:22:37.230562 8689 nvc_info.c:309] missing compat32 library libnvidia-compiler.so
W0605 05:22:37.230592 8689 nvc_info.c:309] missing compat32 library libvdpau_nvidia.so
W0605 05:22:37.230604 8689 nvc_info.c:309] missing compat32 library libnvidia-encode.so
W0605 05:22:37.230628 8689 nvc_info.c:309] missing compat32 library libnvcuvid.so
W0605 05:22:37.230642 8689 nvc_info.c:309] missing compat32 library libnvidia-eglcore.so
W0605 05:22:37.230652 8689 nvc_info.c:309] missing compat32 library libnvidia-glcore.so
W0605 05:22:37.230661 8689 nvc_info.c:309] missing compat32 library libnvidia-tls.so
W0605 05:22:37.230671 8689 nvc_info.c:309] missing compat32 library libnvidia-glsi.so
W0605 05:22:37.230681 8689 nvc_info.c:309] missing compat32 library libnvidia-fbc.so
W0605 05:22:37.230696 8689 nvc_info.c:309] missing compat32 library libnvidia-ifr.so
W0605 05:22:37.230705 8689 nvc_info.c:309] missing compat32 library libGLX_nvidia.so
W0605 05:22:37.230720 8689 nvc_info.c:309] missing compat32 library libEGL_nvidia.so
W0605 05:22:37.230756 8689 nvc_info.c:309] missing compat32 library libGLESv2_nvidia.so
W0605 05:22:37.230767 8689 nvc_info.c:309] missing compat32 library libGLESv1_CM_nvidia.so
W0605 05:22:37.231685 8689 nvc_info.c:331] missing binary nvidia-smi
W0605 05:22:37.231716 8689 nvc_info.c:331] missing binary nvidia-debugdump
W0605 05:22:37.231754 8689 nvc_info.c:331] missing binary nvidia-persistenced
W0605 05:22:37.231774 8689 nvc_info.c:331] missing binary nvidia-cuda-mps-control
W0605 05:22:37.231811 8689 nvc_info.c:331] missing binary nvidia-cuda-mps-server
W0605 05:22:37.231892 8689 nvc_info.c:280] missing ipc /var/run/nvidia-persistenced/socket
W0605 05:22:37.231983 8689 nvc_info.c:280] missing ipc /tmp/nvidia-mps
I0605 05:22:37.232640 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvv4l2.so
I0605 05:22:37.232746 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/desktop-shell.so
I0605 05:22:37.232859 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/drm-backend.so
I0605 05:22:37.232976 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/EGLWLInputEventExample
I0605 05:22:37.233114 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/EGLWLMockNavigation
I0605 05:22:37.233242 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/gl-renderer.so
I0605 05:22:37.233380 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/hmi-controller.so
I0605 05:22:37.233475 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/ivi-controller.so
I0605 05:22:37.233557 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/ivi-shell.so
I0605 05:22:37.233643 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/LayerManagerControl
I0605 05:22:37.233739 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/libilmClient.so.2.2.0
I0605 05:22:37.233820 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/libilmCommon.so.2.2.0
I0605 05:22:37.233893 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/libilmControl.so.2.2.0
I0605 05:22:37.233979 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/libilmInput.so.2.2.0
I0605 05:22:37.234078 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/libweston-6.so.0
I0605 05:22:37.234154 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/libweston-desktop-6.so.0
I0605 05:22:37.234226 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/simple-weston-client
I0605 05:22:37.234305 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/spring-tool
I0605 05:22:37.234395 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/wayland-backend.so
I0605 05:22:37.234499 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston
I0605 05:22:37.234632 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-calibrator
I0605 05:22:37.234723 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-clickdot
I0605 05:22:37.234821 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-cliptest
I0605 05:22:37.235024 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-debug
I0605 05:22:37.235136 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-desktop-shell
I0605 05:22:37.235253 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-dnd
I0605 05:22:37.235342 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-eventdemo
I0605 05:22:37.235430 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-flower
I0605 05:22:37.235600 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-fullscreen
I0605 05:22:37.235696 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-image
I0605 05:22:37.235782 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-info
I0605 05:22:37.235865 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-keyboard
I0605 05:22:37.235951 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-launch
I0605 05:22:37.236030 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-multi-resource
I0605 05:22:37.236126 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-resizor
I0605 05:22:37.236249 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-scaler
I0605 05:22:37.236344 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-screenshooter
I0605 05:22:37.236433 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-simple-dmabuf-egldevice
I0605 05:22:37.236518 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-simple-egl
I0605 05:22:37.236598 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-simple-shm
I0605 05:22:37.236681 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-simple-touch
I0605 05:22:37.236763 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-smoke
I0605 05:22:37.236849 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-stacking
I0605 05:22:37.236932 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-subsurfaces
I0605 05:22:37.237024 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-terminal
I0605 05:22:37.237126 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/weston/weston-transformed
I0605 05:22:37.237216 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvarguscamerasrc.so
I0605 05:22:37.237328 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvcompositor.so
I0605 05:22:37.237413 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvdrmvideosink.so
I0605 05:22:37.237487 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnveglglessink.so
I0605 05:22:37.237563 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnveglstreamsrc.so
I0605 05:22:37.237640 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvegltransform.so
I0605 05:22:37.237715 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvivafilter.so
I0605 05:22:37.237789 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvjpeg.so
I0605 05:22:37.237870 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvtee.so
I0605 05:22:37.237948 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so
I0605 05:22:37.238034 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so
I0605 05:22:37.238115 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideocuda.so
I0605 05:22:37.238199 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideosink.so
I0605 05:22:37.238298 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideosinks.so
I0605 05:22:37.238411 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstomx.so
I0605 05:22:37.238481 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libgstnvegl-1.0.so.0
I0605 05:22:37.238551 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libgstnvexifmeta.so
I0605 05:22:37.238618 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libgstnvivameta.so
I0605 05:22:37.238681 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libnvsample_cudaprocess.so
I0605 05:22:37.238755 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/ld.so.conf
I0605 05:22:37.238832 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/libEGL_nvidia.so.0
I0605 05:22:37.238915 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/libGLESv1_CM_nvidia.so.1
I0605 05:22:37.238986 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/libGLESv2_nvidia.so.2
I0605 05:22:37.239063 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra-egl/nvidia.json
I0605 05:22:37.239138 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1
I0605 05:22:37.239226 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libdrm.so.2
I0605 05:22:37.239327 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libGLX_nvidia.so.0
I0605 05:22:37.239421 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvapputil.so
I0605 05:22:37.239536 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvargus.so
I0605 05:22:37.239659 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvargus_socketclient.so
I0605 05:22:37.239738 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvargus_socketserver.so
I0605 05:22:37.239828 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvavp.so
I0605 05:22:37.239910 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvbuf_fdmap.so.1.0.0
I0605 05:22:37.239980 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so.1.0.0
I0605 05:22:37.240061 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvbufsurftransform.so.1.0.0
I0605 05:22:37.240141 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvbuf_utils.so.1.0.0
I0605 05:22:37.240215 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcameratools.so
I0605 05:22:37.240321 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcamerautils.so
I0605 05:22:37.240522 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcam_imageencoder.so
I0605 05:22:37.240609 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcamlog.so
I0605 05:22:37.240703 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcamv4l2.so
I0605 05:22:37.240796 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcapture.so
I0605 05:22:37.240881 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvcolorutil.so
I0605 05:22:37.240952 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvdc.so
I0605 05:22:37.241062 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvddk_2d_v2.so
I0605 05:22:37.241132 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvddk_vic.so
I0605 05:22:37.241244 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvdla_compiler.so
I0605 05:22:37.241334 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvdla_runtime.so
I0605 05:22:37.241420 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvdsbufferpool.so.1.0.0
I0605 05:22:37.241489 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnveglstream_camconsumer.so
I0605 05:22:37.241577 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnveglstreamproducer.so
I0605 05:22:37.241654 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnveventlib.so
I0605 05:22:37.241742 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvexif.so
I0605 05:22:37.241843 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvfnet.so
I0605 05:22:37.241929 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvfnetstoredefog.so
I0605 05:22:37.242005 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvfnetstorehdfx.so
I0605 05:22:37.242088 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgbm.so
I0605 05:22:37.242178 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_boot.so
I0605 05:22:37.242287 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_camera.so
I0605 05:22:37.242390 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_force.so
I0605 05:22:37.242480 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_generic.so
I0605 05:22:37.242570 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_gpucompute.so
I0605 05:22:37.242672 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_graphics.so
I0605 05:22:37.242752 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_il.so
I0605 05:22:37.242858 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_spincircle.so
I0605 05:22:37.242949 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_tbc.so
I0605 05:22:37.243043 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvgov_ui.so
I0605 05:22:37.243117 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-eglcore.so.32.3.1
I0605 05:22:37.243201 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-egl-wayland.so
I0605 05:22:37.243268 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-fatbinaryloader.so.32.3.1
I0605 05:22:37.243377 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-glcore.so.32.3.1
I0605 05:22:37.243452 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-glsi.so.32.3.1
I0605 05:22:37.243521 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-glvkspirv.so.32.3.1
I0605 05:22:37.243664 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.32.3.1
I0605 05:22:37.243746 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-rmapi-tegra.so.32.3.1
I0605 05:22:37.243859 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvidia-tls.so.32.3.1
I0605 05:22:37.243954 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvid_mapper.so.1.0.0
I0605 05:22:37.244026 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvimp.so
I0605 05:22:37.244106 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvisp_utils.so
I0605 05:22:37.244204 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so
I0605 05:22:37.244324 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvll.so
I0605 05:22:37.244404 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmedia.so
I0605 05:22:37.244482 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmm_contentpipe.so
I0605 05:22:37.244579 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmmlite_image.so
I0605 05:22:37.244685 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmmlite.so
I0605 05:22:37.244780 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmmlite_utils.so
I0605 05:22:37.245279 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmmlite_video.so
I0605 05:22:37.245394 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmm_parser.so
I0605 05:22:37.245492 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmm.so
I0605 05:22:37.245583 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvmm_utils.so
I0605 05:22:37.245704 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvodm_imager.so
I0605 05:22:37.245809 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvofsdk.so
I0605 05:22:37.245895 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvomxilclient.so
I0605 05:22:37.245987 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvomx.so
I0605 05:22:37.246072 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvosd.so
I0605 05:22:37.246152 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvos.so
I0605 05:22:37.246237 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvparser.so
I0605 05:22:37.246339 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvphsd.so
I0605 05:22:37.246414 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvphs.so
I0605 05:22:37.246496 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvpva.so
I0605 05:22:37.246589 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvrm_gpu.so
I0605 05:22:37.246675 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvrm_graphics.so
I0605 05:22:37.246753 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvrm.so
I0605 05:22:37.246838 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvscf.so
I0605 05:22:37.246927 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvtestresults.so
I0605 05:22:37.247011 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvtnr.so
I0605 05:22:37.247092 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvtracebuf.so
I0605 05:22:37.247171 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvtvmr.so
I0605 05:22:37.247252 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvv4l2.so
I0605 05:22:37.247339 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvv4lconvert.so
I0605 05:22:37.247421 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvvulkan-producer.so
I0605 05:22:37.247498 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libnvwinsys.so
I0605 05:22:37.247617 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libsensors.hal-client.nvs.so
I0605 05:22:37.247700 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libsensors_hal.nvs.so
I0605 05:22:37.247797 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libsensors.l4t.no_fusion.nvs.so
I0605 05:22:37.247885 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libtegrav4l2.so
I0605 05:22:37.247968 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libv4l2_nvvidconv.so
I0605 05:22:37.248039 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/libv4l2_nvvideocodec.so
I0605 05:22:37.248113 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/tegra/nvidia_icd.json
I0605 05:22:37.248190 8689 nvc_info.c:154] selecting /lib/firmware/tegra18x/nvhost_nvdec030_ns.fw
I0605 05:22:37.248317 8689 nvc_info.c:154] selecting /lib/firmware/tegra19x/nvhost_nvdec040_ns.fw
W0605 05:22:37.248466 8689 nvc_info.c:403] missing directory /lib/firmware/tegra21x
I0605 05:22:37.249121 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcublas.so.10.2.2.89
I0605 05:22:37.249220 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcublasLt.so.10.2.2.89
I0605 05:22:37.249297 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libnvblas.so.10.2.2.89
I0605 05:22:37.249368 8689 nvc_info.c:154] selecting /usr/include/cublas_api.h
I0605 05:22:37.249446 8689 nvc_info.c:154] selecting /usr/include/cublasLt.h
I0605 05:22:37.249520 8689 nvc_info.c:154] selecting /usr/include/cublasXt.h
I0605 05:22:37.249611 8689 nvc_info.c:154] selecting /usr/include/cublas.h
I0605 05:22:37.249689 8689 nvc_info.c:154] selecting /usr/include/cublas_v2.h
W0605 05:22:37.249771 8689 nvc_info.c:431] missing symlink /usr/loca/cuda
I0605 05:22:37.249919 8689 nvc_info.c:154] selecting /usr/lib/libvisionworks_sfm.so.0.90.4
I0605 05:22:37.249993 8689 nvc_info.c:154] selecting /usr/lib/libvisionworks.so.1.6.0
I0605 05:22:37.250065 8689 nvc_info.c:154] selecting /usr/lib/libvisionworks_tracking.so.0.88.2
I0605 05:22:37.250349 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn.so.8.0.0
I0605 05:22:37.250426 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_ops_infer.so.8.0.0
I0605 05:22:37.250531 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_ops_train.so.8.0.0
I0605 05:22:37.250606 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_adv_infer.so.8.0.0
I0605 05:22:37.250695 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_etc.so.8.0.0
I0605 05:22:37.250790 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_cnn_infer.so.8.0.0
I0605 05:22:37.250865 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_adv_train.so.8.0.0
I0605 05:22:37.250973 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libcudnn_cnn_train.so.8.0.0
I0605 05:22:37.251033 8689 nvc_info.c:375] missing library /usr/include/cudnn_adv_infer.h
I0605 05:22:37.251088 8689 nvc_info.c:375] missing library /usr/include/cudnn_adv_train.h
I0605 05:22:37.251142 8689 nvc_info.c:375] missing library /usr/include/cudnn_backend.h
I0605 05:22:37.251192 8689 nvc_info.c:375] missing library /usr/include/cudnn_cnn_infer.h
I0605 05:22:37.251237 8689 nvc_info.c:375] missing library /usr/include/cudnn_cnn_train.h
I0605 05:22:37.251325 8689 nvc_info.c:375] missing library /usr/include/cudnn.h
I0605 05:22:37.251382 8689 nvc_info.c:375] missing library /usr/include/cudnn_ops_infer.h
I0605 05:22:37.251428 8689 nvc_info.c:375] missing library /usr/include/cudnn_ops_train.h
I0605 05:22:37.251479 8689 nvc_info.c:375] missing library /usr/include/cudnn_version.h
I0605 05:22:37.251670 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_adv_train_v8.h
I0605 05:22:37.251779 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_cnn_infer_v8.h
I0605 05:22:37.251836 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_cnn_train_v8.h
I0605 05:22:37.251921 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_ops_infer_v8.h
I0605 05:22:37.251974 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_ops_train_v8.h
I0605 05:22:37.252025 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_v8.h
I0605 05:22:37.252063 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_version_v8.h
I0605 05:22:37.252104 8689 nvc_info.c:375] missing library /usr/include/aarch64-linux-gnu/cudnn_v8.h
I0605 05:22:37.252186 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_adv_infer.so.8
I0605 05:22:37.252252 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_adv_infer_static_v8.a
I0605 05:22:37.252318 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_adv_train.so.8
I0605 05:22:37.252370 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_adv_train_static_v8.a
I0605 05:22:37.252420 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_cnn_infer.so.8
I0605 05:22:37.252470 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_cnn_infer_static_v8.a
I0605 05:22:37.252528 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_cnn_train.so.8
I0605 05:22:37.252617 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_cnn_train_static_v8.a
I0605 05:22:37.252670 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_ops_infer.so.8
I0605 05:22:37.252727 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_ops_infer_static_v8.a
I0605 05:22:37.252796 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_ops_train.so.8
I0605 05:22:37.252864 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_ops_train_static_v8.a
I0605 05:22:37.252913 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn.so.8
I0605 05:22:37.252979 8689 nvc_info.c:375] missing library /usr/lib/aarch64-linux-gnu/libcudnn_static_v8.a
W0605 05:22:37.253043 8689 nvc_info.c:431] missing symlink /usr/lib/aarch64-linux-gnu/libcudnn_etc.so
W0605 05:22:37.253121 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn
W0605 05:22:37.253169 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_adv_infer_so
W0605 05:22:37.253194 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_adv_infer_stlib
W0605 05:22:37.253216 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_adv_train_so
W0605 05:22:37.253243 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_adv_train_stlib
W0605 05:22:37.253272 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_cnn_infer_so
W0605 05:22:37.253300 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_cnn_infer_stlib
W0605 05:22:37.253373 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_cnn_train_so
W0605 05:22:37.253392 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_cnn_train_stlib
W0605 05:22:37.253467 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_ops_infer_so
W0605 05:22:37.253483 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_ops_infer_stlib
W0605 05:22:37.253528 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_ops_train_so
W0605 05:22:37.253550 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_ops_train_stlib
W0605 05:22:37.253592 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_so
W0605 05:22:37.253617 8689 nvc_info.c:431] missing symlink /etc/alternatives/libcudnn_stlib
I0605 05:22:37.253851 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libnvinfer.so.7.1.0
I0605 05:22:37.253942 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.0
I0605 05:22:37.254028 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libnvparsers.so.7.1.0
I0605 05:22:37.254106 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.7.1.0
I0605 05:22:37.254191 8689 nvc_info.c:154] selecting /usr/lib/aarch64-linux-gnu/libmyelin.so.1.1.0
I0605 05:22:37.254284 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvInfer.h
I0605 05:22:37.254386 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvInferRuntime.h
I0605 05:22:37.254474 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvInferRuntimeCommon.h
I0605 05:22:37.254552 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvInferVersion.h
I0605 05:22:37.254632 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvUtils.h
I0605 05:22:37.254709 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvInferPlugin.h
I0605 05:22:37.254779 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvInferPluginUtils.h
I0605 05:22:37.254850 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvCaffeParser.h
I0605 05:22:37.254942 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvUffParser.h
I0605 05:22:37.255033 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvOnnxConfig.h
I0605 05:22:37.255123 8689 nvc_info.c:154] selecting /usr/include/aarch64-linux-gnu/NvOnnxParser.h
I0605 05:22:37.255457 8689 nvc_info.c:642] requesting device information with opts: ''
I0605 05:22:37.255850 8689 nvc_info.c:660] listing device (null) ((null) at (null))
NVRM version:   (null)
CUDA version:   10.0

Device Index:   0
Device Minor:   0
Model:          Xavier
Brand:          (null)
GPU UUID:       (null)
Bus Location:   (null)
Architecture:   7.2
I0605 05:22:37.256055 8689 nvc.c:314] shutting down library context
I0605 05:22:37.256136 8690 driver.c:191] terminating driver service
I0605 05:22:37.257125 8689 driver.c:231] driver service terminated successfully
```

#### To Do List:
* Try to build another Docker image with CUDA 10.0 and ROS 1.0. Or build L4T 32.4.2 as a RT kernel and flash into AGX Xavier board
* Finish the experiments of profiling the Docker with ROS 1.0 on generic kernel
