# Daily Notes - Akhil Kodumuri

## Week 1

### Monday June 15, 2020
#### Meetings, Out of Work:
* 10am-11:30am First day orientaion

#### Work done
I familiarized my self with Chameleon systems and went through some documentation/tutorials on how to create a reservation, an instance, assign a floating IP, and then ssh into the hardware
#### Issues/Errors/To Do:
- none

### Tuesday June 16, 2020
#### Meetings, Out of work:

#### Work done
I started looking over the Sage project and its architecture
Notes about Sage:
* The goal of the project is to provide scientists a way to collect and evaluate data on site.
* This will be done by using running large scale ml programs on edge computers within the waggle nodes
* Types of nodes - Sage (any node part of SAGE project), Array of Things (cities), Wild Sage (anywhere outside), Sage Blade (Waggle node, can be used in a machine room), and Waggle (means that a node contains Waggle software stack, AI@Edge runtime libraries, and Edge docker containers)
* Software infrastructure - Sage Core Services (essential components for SAGE framework), Waggle Edge Stack (OS image, fetch and perform any task scheduled from SES), Sage Edge Scheduler (Users submit jobs, handles configuration changes, The SES makes all configuration and system update decisions, and queues up changes that can be pushed out to nodes when they contact Beehive), Sage Lambda Triggers (handles all triggers), Cloud Training Software Stack (allows users to run tests on a virtual Waggle before using SAGE)
* Data and Code Repositories - Sage Data Repository (contains all data from sensor and AI@Edge), Edge Code Repository (Library of AI@Edge code that can be run on Waggle software stack)
* Utilities - Virtual Waggle (environment for edge computing code for the Waggle framework), Bench-top Waggle Driver (user can control physical attributes of Waggle node)
* Support infrastructure - Chameleon (large scale testbed that includes cloud controlled hardware), Beehive (cloud endpoint)
