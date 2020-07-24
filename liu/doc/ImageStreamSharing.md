## Comparison of image stream sharing using FFmpeg, ROS2, and RabbitMQ

### FFmpeg

ffserver.conf file:
```
<Stream test1.mpg>

# coming from live feed 'feed1'
Feed feed1.ffm

# Format of the stream : you can choose among:
# mpeg       : MPEG-1 multiplexed video and audio
# mpegvideo  : only MPEG-1 video
# mp2        : MPEG-2 audio (use AudioCodec to select layer 2 and 3 codec)
# ogg        : Ogg format (Vorbis audio codec)
# rm         : RealNetworks-compatible stream. Multiplexed audio and video.
# ra         : RealNetworks-compatible stream. Audio only.
# mpjpeg     : Multipart JPEG (works with Netscape without any plugin)
# jpeg       : Generate a single JPEG image.
# mjpeg      : Generate a M-JPEG stream.
# asf        : ASF compatible streaming (Windows Media Player format).
# swf        : Macromedia Flash compatible stream
# avi        : AVI format (MPEG-4 video, MPEG audio sound)
Format mpeg

# Bitrate for the audio stream. Codecs usually support only a few
# different bitrates.
#AudioBitRate 32

# Number of audio channels: 1 = mono, 2 = stereo
#AudioChannels 1

# Sampling frequency for audio. When using low bitrates, you should
# lower this frequency to 22050 or 11025. The supported frequencies
# depend on the selected audio codec.
#AudioSampleRate 44100

# Bitrate for the video stream
VideoBitRate 2048

# Ratecontrol buffer size
VideoBufferSize 40000

# Number of frames per second
VideoFrameRate 30

# Size of the video frame: WxH (default: 160x128)
# The following abbreviations are defined: sqcif, qcif, cif, 4cif, qqvga,
# qvga, vga, svga, xga, uxga, qxga, sxga, qsxga, hsxga, wvga, wxga, wsxga,
# wuxga, woxga, wqsxga, wquxga, whsxga, whuxga, cga, ega, hd480, hd720,
# hd1080
VideoSize 1920x1080

# Transmit only intra frames (useful for low bitrates, but kills frame rate).
#VideoIntraOnly

# If non-intra only, an intra frame is transmitted every VideoGopSize
# frames. Video synchronization can only begin at an intra frame.
VideoGopSize 12

# More MPEG-4 parameters
# VideoHighQuality
# Video4MotionVector

# Choose your codecs:
#AudioCodec mp2
#VideoCodec mpeg1video

# Suppress audio
NoAudio

# Suppress video
#NoVideo

#VideoQMin 3
#VideoQMax
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

|  	| FPS 	| Delay (s) 	|
|:-:	|:-:	|:-:	|
| FFmpeg 	| 30 	| 8.75 	|
| ROS2 	| 16.0 	| 0.54 	|
| RabbitMQ 	| - 	| - 	|