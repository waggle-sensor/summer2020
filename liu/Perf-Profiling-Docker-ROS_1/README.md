## Performance profiling of Docker with ROS 1.0

### Docker image setup:
 * CUDA: 10.2
 * OpenCV: 3.4.0
 * TensorFlow: 1.15.2
 * ROS: Melodic
 * Python: 3.6.9

Command to pull the Docker image:
```
$ sudo docker pull liangkailiu/plugin-tensorflow-ros:v5
```
### Pipeline of the ROS applications:
![image](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/pipeline.png)

### Profiling tools:
 * docker stats: container system resources utilization profiling
 * perf: ROS profiling
 * top: ROS profiling

### Power modes on Jetson AGX Xavier:
 * 0: MAXN
 * 1: MODE 10W
 * 2: MODE 15W
 * 3: MODE 30W ALL
 * 4: MODE 30W 6CORE
 * 5: MODE 30W 4CORE
 * 6: MODE 30W 2CORE