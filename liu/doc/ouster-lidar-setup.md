## Notes for setting up ouster LiDAR in ROS Melodic

Git ouster_example packages into src and catkin_make.

### Set up static IP with DHCP
```
sudo apt install dnsmasq dhcpcd5
sudo vim /etc/dhcpcd.conf
```
Into dhcpcd.conf file, add next lines:
```
interface eth0
static ip_address=192.168.5.1/24
```
Restart service
```
sudo service dhcpcd restart
```
Let's now backup dnsmasq.conf and create a new one.
```
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.orig
sudo nano /etc/dnsmasq.conf
```
Add next lines to dnsmasq.conf
```
interface=eth0
dhcp-range=eth0,192.168.5.1,192.168.5.20,255.255.255.0,24h
```
Notice that eth0 is the ethernet interface on Raspberry Pi. Now make sure OS1 is not plugged in to ethernet interface and run next commands:
```
ip addr flush dev eth0
sudo ip addr add 192.168.5.1/24 dev eth0
```
Connect os1 and power it on. Then, run:
```
sudo ip link set eth0 up
sudo dnsmasq -C /dev/null -kd -F 192.168.5.50,192.168.5.100 -i eth0 --bind-dynamic
```
After 15 seconds, you can see DHCP negotiation take place. Now get ip assigned to eth0.

Use that ip to run Ouster Example. It would be something like this:
```
roslaunch os1.launch os1_hostname:=192.168.5.92 os1_udp_dest:=192.168.0.5 lidar_mode:=1024x10 viz:=false
```
## Reference
 - https://github.com/ouster-lidar/ouster_example
 - https://github.com/ouster-lidar/ouster_example/blob/master/ouster_ros/README.md
 - https://github.com/ouster-lidar/ouster_example/issues/137
