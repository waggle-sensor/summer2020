#! /usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Copyright (c) 2015 PAL Robotics SL.
Released under the BSD License.

Created on 7/14/15

@author: Sammy Pfeiffer

test_video_resource.py contains
a testing code to see if opencv can open a video stream
useful to debug if video_stream does not work
"""

import sys
import signal
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


def quit(signum, frame):
    print('')
    print('stop fusion')
    sys.exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    # 参数的长度
    if len(sys.argv) < 2:
        print("You must give an argument to open a video stream.")
        print("It can be a number as video device, e.g.: 0 would be /dev/video0")
        print("It can be a url of a stream, e.g.: rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
        print("It can be a video file, e.g.: myvideo.mkv")
        exit(0)

    resource = sys.argv[1]
    # If we are given just a number, interpret it as a video device
    if len(resource) < 3:
        resource_name = "/dev/video" + resource
        resource = int(resource)
    else:
        resource_name = resource
    print("Trying to open resource: " + resource_name)
    cap = cv2.VideoCapture(resource)
    cap.set(3,1920)
    cap.set(4,1080)
    if not cap.isOpened():
        print("Error opening resource: " + str(resource))
        print("Maybe opencv VideoCapture can't open it")
        exit(0)
    bridge = CvBridge()
    publisher = rospy.Publisher('image_raw', Image, queue_size=10)
    rospy.init_node('web_cam')
    print("Correctly opened resource, starting to show feed.")
    rval, frame = cap.read()

    while rval:
        # 将opencv格式的图片转换为ROS可接受的msg
        image_message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_message.header.stamp = rospy.Time.now()
        try:
            publisher.publish(image_message)
        except CvBridgeError as e:
            print(e)
        rval, frame = cap.read()
