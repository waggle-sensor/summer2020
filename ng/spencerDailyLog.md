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

**Monday, June 22**

* Research sampling methods for machine learning and YOLOv3 documentation
* Obtained car images, labels, and masks from Omar
* Began writing Python script to parse and filter annotations in preparation for model training

**Tuesday, June 23**

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

**Wednesday, June 24**

* Ran a test run of the YOLOv3 scripts on the car datasets
* Looked into Stanford car dataset for training instead of Waggle images
  * Wrote script and found out there's at most 100 images per class, too few
* Discussed changing to character recognition data with Nicola and Luke
* Wrote scripts to label and clean Char74K dataset
* Ran test run of YOLOv3 script on local computer (creating a checkpoint after one epoch)
  * Will test on Lambda once I gain access


**Thursday, June 25**

* Had AI/ML scrum meeting
* Attended student seminar on I/O in computing
* Added stratified sampling for characters and filtered used classes based on the most frequent ones
* Attended student cohort session
* Created data set augmentation script for training images
* Gained access to lambda and blues
* To do: run augmentation and train on Lambda

**Friday, June 26**

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

## Week 3

**Monday, June 39**

* Discussed how to move forward with Luke after analyzing results of the character model
  * Recognizes test data with 85% accuracy, but fails to detect individual characters in signs
  * Due to training on pre-segmented images?
* Created bounding box label conversion script for annotated full scene images from 74k character dataset
* Discussed sampling techniques and moving forward with Luke and Nicola
* Implemented interative stratified sampling algorithm for training and testing splits with multilabeled images

**Tuesday, June 30**

* Attended AI/ML scrum meeting
* Checked results of scene character detector trained overnight - mean average precision of 0.11
  * Going to stick with individual (pre-segmented) character classification instead
  * This was after increasing image size to 1000x1000
* Wrote scripts to clean up the KAIST dataset
  * Parse XML annotations into general `Annotation` class created for 74K dataset scenes
  * Function to crop images based on their bounding boxes for sampling and retraining
* To do:
  * Write functions for sampling
  * Create pipeline for retraining after getting detection results

**Wednesday, July 1**

* Checked progress on character training model
* Developed scripts to visualize accuracy over time
  * Visualizer tool to convert Tensorboard output log
  * Modified testing script to output results
* Created presentation on progress thus far
* Fixed bug in augmentation script
* Researched methods for sampling

**Thursday, July 2**

* Attended scrum meeting
* Developed rudimentary (theoretical) model for sampling
  * Generate a probability vs. confidence value curve to determine likelihood of choosing an image for sampling
  * Sample until a quota is met per class
* Presented on my work so far and listened to other presentations
* Attended peer cohort session
* Beginning to create tools to benchmark model performance against KAIST dataset

## Week 4

**Monday, July 6**

* Created script to view benchmark results
  * Histograms for hits, misses, and a combinations
  * On a class-by-class basis or aggregate results
  * Also computes precision/accuracy values and confusion matrix
* Plots are *very* left-skewed for the character model with the 8 most frequent characters
  * High confidence and overall high accuracy -> not much room for improvement
  * See results at [this link](https://drive.google.com/file/d/1Yykr0N00bwvucnBode34G4PR7dPdC4Wj/view?usp=sharing)
* Will retrain on 30-class model, using 84 images per class for training, to determine if it leads to a good benchmark
  * May need to adjust number of classes and samples
  * New parameters: 256x256 image, batch size of 16
  * Preventing class imbalance by undersampling based on the class with the fewest number of images
* Beginning to work on sampling and retraining pipeline

**Tuesday, July 7**

* Attended scrum meeting
* After training 30-class model overnight for 40 epochs, analyzed [results](https://drive.google.com/drive/u/1/folders/18W9wIzQ5cVryueoFM2kgPzl26BI9hFnK)
  * Accuracy remains low: 0.541 on all data, 0.386 when accounting for complete misses
  * Many complete misses compared to 8 class model
    * Could be due to needing more training time
    * Fewer complete misses as time went on
  * Loss curves show large decreases early on
  * mAP is less than 0.10 throughout, possibly due to many bounding boxes being generated
* Next steps: retrain basline model with commonly-confused classes?
* Continued streamlining tools for retraining and sampling, for more general use cases