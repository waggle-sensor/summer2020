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
{'1592417716\n': 2, '1592417717\n': 28, '1592417718\n': 28, '1592417719\n': 28, '1592417720\n': 28, '1592417721\n': 28, '1592417722\n': 28, '1592417723\n': 28, '1592417724\n': 28, '1592417725\n': 29, '1592417726\n': 28, '1592417727\n': 28, '1592417728\n': 28, '1592417729\n': 28, '1592417730\n': 28, '1592417731\n': 28, '1592417732\n': 28, '1592417733\n': 29, '1592417734\n': 28, '1592417735\n': 28, '1592417736\n': 28, '1592417737\n': 28, '1592417738\n': 28, '1592417739\n': 28, '1592417740\n': 28, '1592417741\n': 28, '1592417742\n': 29, '1592417743\n': 28, '1592417744\n': 28, '1592417745\n': 28, '1592417746\n': 4}

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
{'1592419056\n': 2, '1592419057\n': 5, '1592419058\n': 6, '1592419059\n': 5, '1592419060\n': 5, '1592419061\n': 5, '1592419062\n': 6, '1592419063\n': 5, '1592419064\n': 5, '1592419065\n': 5, '1592419066\n': 6, '1592419067\n': 5, '1592419068\n': 5, '1592419069\n': 5, '1592419070\n': 6, '1592419071\n': 5, '1592419072\n': 5, '1592419073\n': 5, '1592419074\n': 6, '1592419075\n': 5, '1592419076\n': 5, '1592419077\n': 5, '1592419078\n': 6, '1592419079\n': 5, '1592419080\n': 1, '\n': 1}
```
The result shows 5 or 6 frames per second.

Ref:
 - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
 - http://wiki.ros.org/cv_bridge/Tutorials
