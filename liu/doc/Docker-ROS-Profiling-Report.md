# Docker ROS Analysis Report

## Overview

With the development of the Internet of things and artificial intelligence, terabytes of data is generated at the edge and needs to be processed in real-time. Edge computing has been proposed to address the latency and bandwidth issues. In this project, we set up experiments with Docker containers and ROS for real-time processing of sensor data. Several ROS applications including object detection, image classification, and handwritten digit classification are deployed to measure the latency, system utilization, and energy consumption of the devices.

## Experiment setup

NVIDIA Jetson AGX Xavier is used to conduct all the experiments. Here are the system configurations of the two Jetson AGX boards used for the experiments. Jetpack is used to flash the host system and we apply RT kernel patches to one of the boards. Inside the AGX board, Docker is used as the running environment for ROS applications.  I built a Docker image with ROS 1.0 running inside.  The versions for CUDA, OpenCV, TensorFlow, and ROS are shown in the above table.

Four ROS applications are set up for the experiments:  YOLO v3 based object detection, MobileNet-SSD v1 based object detection, image classification, and MNIST handwritten digit classification. Besides, several tools are used to collect the logs to measure the latency, CPU/GPU/memory utilization, and energy consumption. The docker stats command is used to get the CPU and memory usage of the running container. NVIDIA Jetson tegrastats is used to collect the usage and frequency of CPU, GPU, memory, etc. 

+---------------------------+---------------+--------------------+
|          Metrics          |  Jetson AGX-1 |    Jetson AGX-2    |
+--------------+------------+---------------+--------------------+
|              |   Kernel   | 4.9.140-tegra | 4.9.140-rt93-tegra |
|     Host     +------------+---------------+--------------------+
|              |   JetPack  |                 4.4                |
+--------------+------------+------------------------------------+
|              |    CUDA    |                10.2                |
|              +------------+------------------------------------+
|              |   OpenCV   |                3.4.0               |
| Docker image +------------+------------------------------------+
|              | TensorFlow |               1.15.2               |
|              +------------+------------------------------------+
|              |     ROS    |            1.0 (Melodic)           |
+--------------+------------+------------------------------------+

The setup notes can be found at: https://github.com/waggle-sensor/summer2020/tree/master/liu. 

## Yolov3 under different power modes

There are seven power modes in AGX Xavier, the detailed descriptions of power budget, online CPU cores, CPU/GPU maximum frequency, GPU TPC (Texture Processor Cluster), and etc are shown in Table 2. We can find the MAXN mode (mode 0) has the best performance.

+--------------------------------+---------------------------------------------------+
|                                |                        Mode                       |
|            Property            +--------+------+-------+------+------+------+------+
|                                |  MAXN  |  10W | 15W * |  30W |  30W |  30W |  30W |
+--------------------------------+--------+------+-------+------+------+------+------+
|          Power budget          |   n/a  |  10W |  15W  |  30W |  30W |  30W |  30W |
+--------------------------------+--------+------+-------+------+------+------+------+
|             Mode ID            |    0   |   1  |   2   |   3  |   4  |   5  |   6  |
+--------------------------------+--------+------+-------+------+------+------+------+
|           Online CPU           |    8   |   2  |   4   |   8  |   6  |   4  |   2  |
+--------------------------------+--------+------+-------+------+------+------+------+
|   CPU maximal frequency (MHz)  | 2265.6 | 1200 |  1200 | 1200 | 1450 | 1780 | 2100 |
+--------------------------------+--------+------+-------+------+------+------+------+
|             GPU TPC            |    4   |   2  |   4   |   4  |   4  |   4  |   4  |
+--------------------------------+--------+------+-------+------+------+------+------+
|   GPU maximal frequency (MHz)  |  1377  |  520 |  670  |  900 |  900 |  900 |  900 |
+--------------------------------+--------+------+-------+------+------+------+------+
|            DLA cores           |    2   |   2  |   2   |   2  |   2  |   2  |   2  |
+--------------------------------+--------+------+-------+------+------+------+------+
|      DLA maximal frequency     | 1395.2 |  550 |  750  | 1050 | 1050 | 1050 | 1050 |
+--------------------------------+--------+------+-------+------+------+------+------+
|            PVA cores           |    2   |   0  |   1   |   1  |   1  |   1  |   1  |
+--------------------------------+--------+------+-------+------+------+------+------+
|      PVA maximal frequency     |  1088  |   0  |  550  |  760 |  760 |  760 |  760 |
+--------------------------------+--------+------+-------+------+------+------+------+
| Memory maximal frequency (MHz) |  2133  | 1066 |  1333 | 1600 | 1600 | 1600 | 1600 |
+--------------------------------+--------+------+-------+------+------+------+------+

### end-to-end latency

For YOLO v3 based object detection, the message pipeline of the data sharing demo in ROS is shown in Figure 1. A router is used to collect devices because the transmission of ROS messages across multiple machines requires them to be under the same LAN.

The whole application starts with a ROS node /usb_cam capturing images from the USB camera, then images will be published to the system using ROS messages (called /usb_cam/image_raw). Another ROS node called /darknet_ros is launched which subscribes to the image topic and executes model inference of YOLO v3 model for object detection. The detection results as well as the raw images are published out for the access of other processes or other machines. 

![image](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/pipeline.png)

End-to-End latency is defined as the total time consumption from the start time of capturing an image frame to the end time of getting detection results and publishing the results out. Here we repeated the experiments under different power modes and the CDF of the end-to-end latency is shown in Figure 3. We can find that power mode 0 has the best performance compared with others. 

CDF with Power Modes             |  CDF with USB camera and image file
:-------------------------:|:-------------------------:
![](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/yolov3-latency-cdf.png)  |  ![](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/mode0_cdf_usbcam.png)