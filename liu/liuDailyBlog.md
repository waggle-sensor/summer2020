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
