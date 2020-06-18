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
* 1pm meeting meeting with Jake and Kate
* 4:30pm - 5:10pm meeting with Raj and Jake
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



Notes about Docker
* Docker places processes that are running on a system into a container. A container runs the process in its own environment and contains its own file system.

I went over the learning agreement with Raj and Jake and learned the project goals and what is expected of me.


#### Issues/Errors/To Do:
- Continue familiarizing myself with Sage, Docker, and Chameleon
- Figure out why I cannot connect to the Sage Chameleon project

### Wednesday June 17, 2020
#### Meetings, Out of work:
- 11am - 12am Weekly Seminar Meeting
#### Work done
I watched two videos reguarding the Agile Scrum process
Notes on Agile Scrum:
* very collaborative and adapting workflow
* Scrum is not a waterfall project. No dividing work into phases (design, code, etc)
* Scrum has "sprint" phases
* every sprint is about 2 weeks and a demo is presented at the end of each sprint
* teams that use Scrum should be diverse
* 3 types of people in scrum teams: product owner, scrum development team, and scrum master
* Product Backlog - List of things to do from product owner -> goals of current sprint
I went through tutorials that physically introduced me to Chameleon and Docker. It turns out the previous days I wasn't fully added to Chameleon project. So now I added my own Application Credential. I also used openstack to connect my laptop to Chameleon.
I downloaded and configured docker and docker compose on my machine.

#### Issues/Errors/To Do:
- Connect my computer to bee-hive server and setup Virtual Waggle

### Thurday June 18, 2020
#### Meetings, Out of Work:
none
#### Work done
I worked on creating a local connection to beehave and configured virtual waggle. In this process I researched Docker and Docker-Compose. I continued to research on how to create multiple Docker containers with docker-compose.  
#### Issues/Errors/To Do:
- connect and create multiple instances of virtual waggle within docker containers on the cloud 
