#!/usr/bin/env python
import pika
import sys
import cv2
import base64
import time

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.exchange_declare(exchange='logs', exchange_type='fanout')

cap = cap = cv2.VideoCapture("/dev/video1")
cap.set(3,1920)
cap.set(4,1080)

ravl, frame = cap.read()

while ravl:
    message = base64.b64encode(frame)
    #print(message)
    channel.basic_publish(exchange='logs', routing_key='', body=message)
    #print(" [x] Sent %r" % message)
    t = time.time()
    print(t)
    ravl, frame = cap.read()

#message = ' '.join(sys.argv[1:]) or "info: Hello World!"
#channel.basic_publish(exchange='logs', routing_key='', body=message)
#print(" [x] Sent %r" % message)
connection.close()
