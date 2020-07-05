#!/bin/bash
sudo docker stats --format "{{.MemUsage}}\t{{.CPUPerc}}\t{{.MemPerc}}" >> docker_stats.log
#top -b -d 1 -n 100 | grep darknet_ros | tee -a top_darknet.log
#top -b -d 1 -n 100 | grep usb_cam_node | tee -a top_usb_cam.log
sudo /usr/bin/tegrastats > tegrastats.log
