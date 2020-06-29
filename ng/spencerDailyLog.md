# Daily Log - Spencer Ng

## Week 1

**Monday, June 15**

* Attended new employee orientation
* Set up Argonne email, Slack, and Teams accounts
* Completed safety and other training courses
* Set up local ssh environment for connecting to CELS, MCS, and LCRC servers
* Installed Docker on local computer

**Tuesday, June 16**

* Attended second day of orientation and small group check-in
* Finished remaining courses in TMS
* Attended sprint retrospective meeting
* Looked at SAGE and Waggle component documentation in various repositories
* Attended workshop on scientific writing
* Began reading/watching materials on Agile Scrum
* Set up accounts on Miro and Jira

**Wednesday, June 17**

* Attended workshop for summer students
* Met with Sean and Raj to discuss my background and specific research projects for the summer
* Finished reading/watching documentation on Agile Scrum and Docker
* Reviewed example scientific applications of Sage 
* Reviewed past and current features on Jira and Miro

**Thursday, June 18**

* Began following Docker tutorials on creating/deploying images/containers
* Attended scrum meeting and student peer group session
* Reviewed documentation on Kubernetes and ROS 2.0

**Friday, June 19**

* Attended R&D meeting for AI/ML
* Attended training for workplace safety
* Attended scrum meeting
* Read articles on AoT sensor application
* Met with Raj, Nicola, and Luke on ML model training and sampling model goals
* Discussed how to move forward on the project with Luke
  * Began working on image classifier based on the Waggle plugin example
  * Asked for car dataset from Omar

## Week 2

**Monday, June 21**

* Research sampling methods for machine learning and YOLOv3 documentation
* Obtained car images, labels, and masks from Omar
* Began writing Python script to parse and filter annotations in preparation for model training

**Tuesday, June 22**

* Created bounding box functions with OpenCV to verify annotation accuracy
* Finished YOLOv3 text file label generation for main gate dataset
* Obtained second dataset with make/model (also from Waggle)
  * Wrote parser for second set's labels
* Attended scientific writing seminar
* Documented series of goals for the process:

1. Label datasets of images with make/model and prepare in a format for YOLOv3
2. Split up dataset for training, sampling, and testing (thinking of doing stratified sampling)
3. Train ML model with YOLOv3 with selected images (and run against testing set to create a baseline)
4. Run ML model on second set of images, using different sampling methods to choose images for retraining
5. Retrain ML model
6. Run ML model on testing images, seeing which sampling methods give the best results

* Discussed steps going forward with Raj about potential datasets
* Finished labeling scripts to generate file lists for make/model

**Wednesday, June 23**

* Ran a test run of the YOLOv3 scripts on the car datasets
* Looked into Stanford car dataset for training instead of Waggle images
  * Wrote script and found out there's at most 100 images per class, too few
* Discussed changing to character recognition data with Nicola and Luke
* Wrote scripts to label and clean Char74K dataset
* Ran test run of YOLOv3 script on local computer (creating a checkpoint after one epoch)
  * Will test on Lambda once I gain access


**Thursday, June 24**

* Had AI/ML scrum meeting
* Attended student seminar on I/O in computing
* Added stratified sampling for characters and filtered used classes based on the most frequent ones
* Attended student cohort session
* Created data set augmentation script for training images
* Gained access to lambda and blues
* To do: run augmentation and train on Lambda

**Friday, June 25**

* Attended chat with Mike Papka
* Fixed bugs in augmentation script and deployed it on Lambda and Blues
* Wrote sbatch script to deploy on Lambda/Blues and set up Anaconda environment/dependencies
  * Job couldn't be submitted on Lambda, Blues undergoing maintenance this weekend
* Searched for other datasets for sampling and retraining
  * Google Street View has 300 images, has no character-specific annotations
  * KAIST has one with bounding boxes for characters and similar distributions to our training dataset, good candidate
    * Wrote script to determine this
* Launched Chameleon instance for training
* Began reading papers on YOLO(v3) implementation and research on good training data for ML