## Comparison of image stream sharing using FFmpeg, ROS2, and RabbitMQ

### FFmpeg

ffserver.conf file:
```
<Stream test1.mpg>

# coming from live feed 'feed1'
Feed feed1.ffm

Format mpeg
VideoBitRate 2048
VideoBufferSize 40000
VideoFrameRate 30
VideoSize 1920x1080
VideoGopSize 12
NoAudio
VideoQMin 1
VideoQMax 35

</Stream>
```
Launch ffserver and streamming:
```
ffserver &
ffmpeg -r 30 -s 1920x1080 -f v4l2 -i /dev/video0 http://localhost:8090/feed1.ffm
```
Access the video stream:
```
ffplay -i http:localhost:8090/test1.mpg
```

FPS            |  Delay for the image stream
:-------------------------:|:-------------------------:
![](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/ffmpeg-fps.jpeg)  |  ![](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/ffmpeg-latency.jpeg)

### ROS2

Publish image frames:
```
ros2 run image_tools cam2image 0
```

Subscribe to image frames:
```
ros2 run image_tools showimage
```

FPS            |  Delay for the image stream
:-------------------------:|:-------------------------:
![](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/ros2-fps.jpeg)  |  ![](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/ros2-latency.jpeg)

### Results

#### FPS and Delay

One subscriber case:
|  	| FPS 	| Delay (s) 	|
|:-:	|:-:	|:-:	|
| FFmpeg 	| 30 	| 8.75 	|
| ROS2 	| 16.0 	| 0.54 	|
| RabbitMQ 	| 16.0	| 0.103	|
