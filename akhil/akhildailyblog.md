beginnersvariantChameleon-SageSage-Chameleonpresentationpresentation# Daily Notes - Akhil Kodumuri

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

    sudo groupadd docker
    sudo usermod -aG docker cc
    newgrp docker


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

Notes On Kubernetes:
Katacoda: used to run virtual terminal
Minicube: small scale local deployment of Kubernates
Kubernetes coorindates a highly available cluster of computers that are connected to work as a same unit. Kubernetes automates the distribution and scheduling of application containers across a cluster in a more efficient way.
- Master: Coordinates and manages clusters, maintaining node processes and states, and rolls out updates.
- Nodes: are the workers that run the applications. They are VM or computers. Hosts the running applications. Each node has a Kubelet.
- Kubelet: an agent used to manage the node and communicate with Kubernetes master.
Nodes communicate with master using kubernetes api.


#### Issues/Errors
- mcs email account

### Wednesday July 1, 2020
#### Meeting, Out of Work
- 10:30am - 11am

#### Work done
Notes On Kubernetes:
Katacoda: used to run virtual terminal
Minicube: small scale local deployment of Kubernates
Kubernetes coorindates a highly available cluster of computers that are connected to work as a same unit. Kubernetes automates the distribution and scheduling of application containers across a cluster in a more efficient way.
- Master: Coordinates and manages clusters, maintaining node processes and states, and rolls out updates.
- Nodes: are the workers that run the applications. They are VM or computers. Hosts the running applications. Each node has a Kubelet.
- Kubelet: an agent used to manage the node and communicate with Kubernetes master.
Nodes communicate with master using Kubernetes api. It also manages the pods and containers inside of the node.
- Pod: Abstraction that represents the one or more application containers

```
To assign role to node -> kubectl label node node_name node-role.kubernetes.io/worker=worker
```

- Pods are assigned worked by master (master schedules tasks and operations)

#### Issues/Errors
none

### Thursday July 2, 2020
#### Meetings
none
#### Work done
I worked on uploading the Sage-Chameleon image the I created and uploading it onto Chameleon for others to use. I also worked through the errors that occurred when trying to create and upload the image. I also continued working on creating the K8 cluster on a bare metal instance that will demo the deployment of a Sage-Virtual waggle pod.


```
create a container on cluster -> microk8s kubectl run sage --image=Chameleon-Sage-Image-Builder/tripleo-image-elements:v1
```
#### Issues/Errors
none

### Monday July 6, 2020
#### Meetings/Out of work
- 10am - 11:15am -> Weekly Cohort and Cluster meeting
- 11:15am - 1pm -> Scrum meeting and Backlog refinement

#### Work done/Things learned
I am building a Chameleon bare-metal instance with the Chameleon-Sage image with the additions that I made to it. From there, I will make it into a Kubernetes node to be assigned to another bare-metal instance (which will be master). I also started working on my presentation for tripod presentation on Friday and a presentation for the Demo on Friday.

#### Errors/Problems

### Tuesday July 7, 2020
#### Meetings/Out of work
11am - 11:30am -> Daily scrum meeting
1pm - 2:30pm -> Meeting with Jake
#### Work done/Things learned
I worked on fixing the error reguarding the Sage-Chameleon image. Jake and I talked on call and fixed the some more errors, and we are now able to create a virtual waggle session.


#### Errors/Problems
- deploying beehive

### Wednesday July 8, 2020
#### Meetings/Out of work
10:30am - 11am -> Daily Scrm meeting
11am - 12pm -> Weekly speaker series

#### Work done/Things learned
Jake and I ironed out the final error regarding the Sage-Chameleon image. The image can now have a user cc run docker commands without sudo, deploy beehive, and create a virtual waggle instance. This image will be used within a pod in the Kubernetes cluster I will create. I also got approved to merge my branch to the master branch of Chameleon-Sage-Image-Builder. Sage users now have detailed steps on creating their own Chameleon-Sage image. I also worked on making a test deloyment using microk8s, which I will use to learn more about making my own deployments.
###### Notes on Kubernates Deployment
.yaml contains 4 types of specificatons
- apiVersion, kind, metadata, and spec
apiverion -> api version of the kubernetes deployment
kind -> type of object to be created: deployment, pod, nodes, etc
metadata -> set of data to uniquely identify a kubernetes object
spec -> where we declare the desired state and characteristic of the object we have
spec has 3 subfields:
1. Replicas -> make sure the number of pods running all the time for the deployment
2. It defines the labels that match the pods for the deployments to manage.
3. xIt has its own metadata and spec. Spec will have all the container information a pod should have. Container image info, port information, ENV variables, command arguments etc.


###### Personal Non-technical notes on Kubernetes deployments
- namespace is like a group that holds all the parameters/characteristic/restrictions of a deployment


#### Errors/Problems

### Thursday July 9, 2020
#### Meetings/Out of work
11am-11:30am -> Daily Scrum meeting

#### Work done/ Things Learned
I worked one and finished my cluster presentation. I continued working on the YAML script to launch a k8 cluster to deploy instances of virtual waggle.

#### Errors/Problems

### Friday July 10, 2020
#### Meetings/Out of work
10am - 11am -> Cluster presentaions
11am - 11:30am -> Daily Scrum Meeting
3pm-5:30pm -> Demos

#### Work done/Things learned
I demoed my progress that I made this sprint and gave my presentation to my cluster group.

#### Errors/Problems

### Monday July 13, 2020
#### Meeting/Out of work
10am-10:30am -> Weekly Cohort Meeting
11am - 1pm -> Scrum sprint planning
#### Work done/Things Learned
I noticed I am having some troubles setting up the YAML for the k8 deployment, so I did some more research into creating this file.
Note for later: imagePullPolicy in Kubernetes
Notes on YAML deployment:
-

#### Errors
- none

### Tuesday July 14, 2020
#### Meeting/Out of work
9:30am - 10am Meeting in AI scrum
11am - 11:30am -> Daily srum meeting
2pm -> 3pm -> Meeting with Sean
#### Work done/Things learned
I cotinued working on the k8 deployment. I am having trouble with pulling the images and the scope of the deployment. I have meeting with Sean that will hopefully help.
#### Errors
- Image pull error

#### Wednesday July 15, 2020
10:30am -> 11am Daily Scrum meeting
11am -> 12:30pm weekly seminar
#### Work done/Things learned
I found a resource that can translate docker compose files to kubenetes deployment files. It's called ``` kompose  ```. I then went through the files created and troubleshooted the errors because there were a lot of them.

#### Error
- Volumes, restartPolicy, and network configuration in k8 deployment

### Thursday July 16, 2020
#### Meetings/Out of work
- none
#### Work done/Things learned
I continued to debug the errors. I was able to correctly configure the virtual-waggle enviorment. However, there are still a lot of error and no containers are able to run.

#### Errors
- Getting containers to run

### Friday July 17, 2020
#### Meetings/Out of work
1:30-1:45 -> meeting with cohort leader
#### Work done/Things learned
I started working on the presentation I have to give on Monday. I also was able to figure out the volumes issue and get the rabbitmq container. I am having trouble configuring the network for the containers. I will try to see if going into the contianers of the deployment to see if that helps

#### Errors
- network config


### Monday July 20, 2020
#### Meetings/Out of work
10am - 11am -> cohort lightning talk
11am - 12:30pm -> Scrum backlog refinement
3pm - 4pm -> meeting with Sean and Jake
#### Work done/Things learned
I had meetings with Sean and Jake and we talked about how the k8 deployment is not scalable with the current state of virtual-waggle. Thus, I worked on figuring out what to do next. I then created a script in the Image building process that clones the edge-plugin repo for a user to use.
#### Errors
- none

### Tuesday July 21, 2020
#### Meetings/Out of work
10:30am - 11am -> meeting with Raj and Sean
11am - 11:30am -> Daily scrum meeting
#### Work done/Things learned
I created a pull request for the new bash script I made. I also worked on building the Chameleon-Sage image for a virtual-machine. I also started researching github actions. I will be creating a github action for building the Chamaleon-Sage image.

#### Errors
- none

### Wednesday July 22, 2020
#### Meetings/Out of work
10:30am - 11am -> Daily scrum meeting
11am -> 12pm - profesional development workshop
#### Work done/Things learned
I worked on learning and practicing runs and builds on github actions. There isn't much documentation on github actions and its setup for begininers, so it took some time for me to learn how to create, build, and test an action. I created a new branch for testing the image building process. Each test will take a long time since there are 3 sites and each image takes about 5-10 mins to build. Now I will start on finding a way to upload the image to all three sites.
#### Errors/issues
- creating GITHUB_SECRETS and environment for the rc file

### Thurday July 23, 2020
#### Meetings/Out of work
11am - 11:30am -> Daily Scrum meeting

#### Work done/Things learned
I continued researching on an effective way to upload images and building them in github actions. I also need to do research on creating environments in github actions. This is a specfic enviorment must be configured to show valid credentials. I also figured out a way to create a secret on Github. I will encript my Openstack/Chameleon username and password.
#### Errors/issues
- Creating an environment for upload
- uploading image to Chameleon

### Friday July 24, 2020
#### Meetings/Out of work
11am - 11:30am -> Daily Scrum meeting
3pm - 4:30pm -> Sage demo

#### Work done/Things learned
I continued working on uploading images to Chameleon. I also prepared what I will demo at the scrum meeting. I am goint to demo how to create a VM at the KVM@TACC Chameleon site.

#### Errors/Issues
none

### Monday July 27, 2020
#### Meetings/Out of work
10am - 11am -> Weekly cohort meeting
11am - 1pm -> scrum planning

#### Work done/Things learned
I worked with Jake to figure out my stories for next sprint. I will work on cleaning up the Github action I created ti upload images to each Chameleon site. I will also find a way to create Chameleon-Sage images with cuda and arm64 hardware specifications. I will also work on adding the ECR to the Chameleon-Sage image and backdating old versions of the Chameleon-Sage image in the Openstack cloud

#### Errors/Issues
none

### Tuesday July 28, 2020
#### Meetings/Out of work
11am - 11:30am -> Daily scrum meeting

#### Work done/Things learned
I worked on restructuring the Github action to the appropriate name changes and researched on how to back date images on Chameleon. I added code to create-image.py to rename the images.

#### Errors/Issues
none

### Wednesday July 29, 2020
#### Meetings/Out of work
10:30am - 11:00am -> Daily scrum meeting
11am - 12pm -> Weekly speaker series
#### Work done/Things learned
I finished changing the naming conventions to the Chamaleon-Sage images. I also created two new variants that users can create for Sage images. One vairant is for CUDA specifications and the choice of using arm64. I also worked on publishing these images to Chameleon's catalog. I also continued working on creating a backdating script.

#### Errors/Issues
- how to create a professional image writeup
