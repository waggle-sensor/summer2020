## ROS CV_Bridge Delay

ROS passes around images in its own sensor_msgs/Image message format, but many users will want to use images in conjunction with OpenCV. CvBridge is a ROS library that provides an interface between ROS and OpenCV. [CvBridge](http://wiki.ros.org/cv_bridge) can be found in the cv_bridge package in the vision_opencv stack.

![image](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/cvbridge.png)

### FPS without cv_bridge
```
cap = cv2.VideoCapture("/dev/video1")
cap.set(3,1920)
cap.set(4,1080)
while rval:
        rval, frame = cap.read()
```

Number of frames for each second:
```
{'1592417716': 2, '1592417717': 28, '1592417718': 28, '1592417719': 28, '1592417720': 28, '1592417721': 28, 

'1592417722': 28, '1592417723': 28, '1592417724': 28, '1592417725': 29, '1592417726': 28, '1592417727': 28, 

'1592417728': 28, '1592417729': 28, '1592417730': 28, '1592417731': 28, '1592417732': 28, '1592417733': 29, 

'1592417734': 28, '1592417735': 28, '1592417736': 28, '1592417737': 28, '1592417738': 28, '1592417739': 28, 

'1592417740': 28, '1592417741': 28, '1592417742': 29, '1592417743': 28, '1592417744': 28, '1592417745': 28, 

'1592417746': 4}
```
The result shows 28 or 29 frame per second.

### FPS with cv_bridge
```
cap = cv2.VideoCapture("/dev/video1")
cap.set(3,1920)
cap.set(4,1080)
  
bridge = CvBridge()
publisher = rospy.Publisher('image_raw', Image, queue_size=10)
rospy.init_node('web_cam')
rval, frame = cap.read()

while rval:
    image_message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    image_message.header.stamp = rospy.Time.now()
    try:
        publisher.publish(image_message)
    except CvBridgeError as e:
        print(e)
    rval, frame = cap.read()
```

Number of frames for each second:
```
{'1592419056': 2, '1592419057': 5, '1592419058': 6, '1592419059': 5, '1592419060': 5, '1592419061': 5, 

'1592419062': 6, '1592419063': 5, '1592419064': 5, '1592419065': 5, '1592419066': 6, '1592419067': 5, 

'1592419068': 5, '1592419069': 5, '1592419070': 6, '1592419071': 5, '1592419072': 5, '1592419073': 5, 

'1592419074': 6, '1592419075': 5, '1592419076': 5, '1592419077': 5, '1592419078': 6, '1592419079': 5, 

'1592419080': 1}
```
The result shows 5 or 6 frames per second.

Ref:
 - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
 - http://wiki.ros.org/cv_bridge/Tutorials
