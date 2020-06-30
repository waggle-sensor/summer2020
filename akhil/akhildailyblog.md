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
I worked on creating a local connection to beehive and configured virtual waggle. In this process I researched Docker and Docker-Compose. I continued to research on how to create multiple Docker containers with docker-compose. I started working on why the registration container isn't responding.

Things I tested:
sudo docker update --restart=no <container_id>
docker logs --tail 50 --follow --timestamps container_id

#### Issues/Errors/To Do/Questions:
- connect and create multiple instances of virtual waggle within docker containers on the cloud
- registration container always restarting

### Friday June 19, 2020
#### Meeting, Out of Work:
none
#### Work done
In order to fix the issue of the constantly restarting registration container, I ran the command ```docker run --rm -it --env-file "./waggle-node.env" -v "${WAGGLE_ETC_ROOT}:/etc/waggle" waggle/registration /bin/sh```. This creates the registration container, but I am able to access the terminal within the container. From here I can debug ``` registration.py ``` by using ``` ping host.docker.internal ```. The debugged

#### Issues

### Monday June 22, 2020
#### Meeting, Out of Work:
- 10am-10:30am
#### Work done
I worked on creating a bare-metal instance that uses 18.04 Ubuntu image that can be configured to work with beehive and run virtual waggle. I also worked with numerous issues with creating an instance, ssh key, key pair, private, and public key and trying to ssh into the instance from command line.

#### Issues/Errors
- not enough floating IPs
- Error: Permission denied (publickey)

### Tuesday June 23, 2020
#### Meeting, Out of Work:
- 9am - 1pm Mandatory Carpentry Workshop - Advanced Python

#### Work done
I attended the virtual workshop on Advanced python programming. I learned about error catching, defensive programming, and using the Pandas library for many different purposes: Reading, writing, graphing, displaying csv files.
I was also able to build an Ubuntu image on run it on a Chameleon bare-metal instance. I also resolved yesterdays issues and sshed into the instance. I then started downloading an Ubuntu image from the cloud in order to create an image that can be used in Docker containers.

#### Issues, Errors
- Documentaion is a little confusing and I am getting issues with authentication

### Wednesday June 24, 2020
#### Meeting, Out of Work:
- 9am - 1pm Mandatory Carpentry Workshop - Advanced Unix shell commands

#### Work done
I worked through the issues from yesterday with the assistance of Jake. I figure that I was running the python script incorrectly.
Correct way: ```python create-image.py --release bionic --variant sage --region CHI@UC ```
Then I had to work through the ``` glance ``` command that is supposed to register the image to the open stack system. I had to work through some of these issues as well.
Correct glance command: ``` glance image-create --name "CC-Ubuntu18.04-SAGE" ```

#### Issues/Errors
- variant flag error when running ``` glance ``` command

### Thursday June 25, 2020
#### Meeting, Out of Work:
- 9am - 1pm Mandatory Carpentry Workshop - Advance Git

#### Work done
I worked through yesterdays variant error. I had to check out the sage branch, I can use the sage as a variant
Check out: ```git fetch && git checkout sage;```
Quick recap of what I have done so far: Chameleon-Sage-Image-Builder downloads an Ubuntu image then customizes it so it can be used on the Chameleon bare-metal instance and then, ultimately, generates an image with Virtual-Waggle that can interact with beehive. I then started working on creating a docker group that can deploy beehive and run virtual-waggle without admin privledges (without sudo). I started to research how to do this

#### Issues/Errors
- how to create docker group with user that doesn't need to use sudo to ```waggle-node up ``` and ``` beehive deploy ```

### Friday June 26, 2020
#### Meeting, Out of Work
- 9am - 1pm Mandatory Carpentry Workshop - Advance SQL

#### Work done
I continued researching on how to create a docker group with user cc. cc should be able to deploy beehive and run virtual waggle without sudo. I was able to create docker group and added user cc to it.
Code I used
``` sudo groupadd docker
    sudo usermod -aG docker cc
    newgrp docker
                    ```
However, when trying to run the commands I still receive an error.

#### Issues/Errors
- Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/containers/json: dial unix /var/run/docker.sock: connect: permission denied

### Monday June 29th, 2020
#### Meeting, Out of Work
- 10am-10:45am Weekly Cohort Standup
- 11:30am - 1pm Scrum meeting
- 3pm - 3:45pm Meeting with Jake

#### Work done
I worked through yesterdays error and successfully created a docker group where user cc doesn't need sudo to run docker commands. I then started to add a section to the markdown on Chameleon-Sage-Image-Builder that will help Sage users replicate my process. I will also start to research Kubernates.
#### Issues/Errors
- I need access to make commits to the Chameleon-Sage-Image-Builder repo

### Tuesday June 30th, 2020
#### Meeting, Out of work

#### Work done
I added steps that a Sage user can take to create a Chameleon instance and create a Chameleon-Sage image. I also added steps on how to create a docker group with user cc that can use Docker commands without ```sudo```

#### Issues/Errors
