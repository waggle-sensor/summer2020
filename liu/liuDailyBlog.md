# Daily Notes - Liangkai Liu

## Week 1 (May 4 to May 8)

### May 4 to May 7, 2020
#### Work Done:

**Prototype Setup: **set up a prototype with a Leopard HD camera and a Velodyne VLP16 LiDAR. 
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
