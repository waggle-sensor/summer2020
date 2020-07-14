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

**Wednesday, July 8**

* Attended student seminars for scientific presentations
* Met with Nicola to discuss how to move forward with creating a better baseline
* Examined confusion matrix of characters for poor fits
* Criteria for poor fit: 10% or more of an actual class were predicted to be an alternative class
  * Normalized ratio, ignoring any images with no object detection
* Poor fits:
  * B: E, e
  * D: P, e
  * E: F
  * I: L, l, t
  * L: l
  * M: K, V
  * N: K
  * O: e, o
  * P: t
  * R: H
  * S: e, s
  * T: t
  * U: P
  * a: e
  * i: L, l, t
  * l: L, r
  * o: e, t
  * r: l, t
  * s: S, e
  * t: L
* Best fits (30%+ predicted to be actual class): A, I, K, P, R, V, W, e, t
* Final list of 12 classes: B, E, F, P, e, D, L, I, t, K, S, O
* Began training new model
  * Batch size: 32
  * Image size: 256x256
  * Still using undersampling, this time using 100/34 train/test split
* Tested new model and analyzed results
  * Higher accuracy/precision as training went on
  * New issue: not detecting any objects, leading to low mAP
* Trying to retrain with the same classes/splits, using new image augmenting method to see if results improve
  * Using one "major" transformation (shift, distort, crop, etc.) combined with 1-2 "minor" transformations (RGB shift, noise, blur, etc.)
  * Batch size modified to 64 for reduced processing time per epoch
  * Modified image size to 416x416, after reading that YOLO works better on small objects with higher resolutions
  * Created 120 augmentations per image
* Wrote some documentation on using the KAIST scripts


**Thursday, July 9**

* Attended scrum meeting
* Benchmarked results from augmented 12-class model ran overnight
  * mAP of 0.75 after 100 epochs
  * Converged after 50 epochs
  * Going to use as a baseline
* Attended seminar on climate change modeling
* Attended Sage development tutorial
* Added test set mAP to custom benchmark tool
* Continued implementing sampling methods, accounting for class imbalance via undersmapling
* Completed label generation for new training/sampled files
  * Bit of a hack, as this doesn't create a new `labels` folder
* To do: develop actual pipeline for sampling
  * Thinking of executing one command per sampling method (`python3 sample.py <method>`)
  * Need to rename testing file, accounting for different data paths
  * Autogenerate data config
  * Custom naming for checkpoint weights to differentiate
  * Need to call `augment.py` and a custom version of `train.py`
    * Modularizing functions there even more might help
  * Rely on existing files in `config`

**Friday, July 10**

* Wrote pipeline for benchmarking, sampling, and retraining
  * Need to modularize more code so we don't rely on system calls for training and augmentation
  * Many constants still programmed in - need to decide what to keep
* Running simple retraining test using a median threshold sampling
  * No undersampling used
  * Hit rate of sample: 0.9840182648401826
  * 1741 raw training images in total (roughly half of sample)
* Used augmentation techniques on new training set, keeping the old testing set
  * Instead of running the composition function 120 times per image, we set a scale factor based on the number of images in the inferred class
  * This prevents class imbalance without having undersampling
  * Technique could be useful for batch processing in the future 
  * Targeting 15,000 images per class
  * Numbers are slightly off due to rounding scale factors early on in the process and having a set number of augmentations per image in a particular class
  * Roughly 100-150 transformations per image
* To do
  * Modularize code more in preparation for dockerizing 
  * Investigate more sampling methods - optimize hit rate while having hard cases

## Week 5

**Monday, July 13**

* Analyzed results from median threshold resampling technique
  * Against the test set, precision (with a hard classification, not mAP) decreased from benchmark of 0.862 around 0.81, depending on the epoch
  * Against the sample set, precision increased from 0.859 to peak of 0.893 at epoch 103, then dropped and hovered around 0.875
  * Large number of augmented images per class from samples used for training might have led to overfitting occurring quickly
* Retraining has more important implications for improving detection at environments where sampling occurred, not the original test data
  * I.e. opposite of generalization
  * Could still be useful if Sage nodes are deployed to a set of specific environments and trained on generic data (e.g. COCO)
  * A for of continuous, unsupervised learning
* Further tests:
  * Lowering augmentation (images per class) parameters to prevent overfitting
  * Lowering proportion ("bandwidth problem") of images that are sampled, instead of 100%
  * Normalizing precision to be the mean of precision by class
  * Iterative retraining? Could split up KAIST data further
  * Actual object detection
* Generated plot of precision vs. confidence range, general positive trend
* Currently retraining models based on new sampling methods with more "hard" cases:
  * **IQR**: samples a proportion (currently 100%) of the data within the interquartile range
  * **Normal distribution**: assumes a Normal distribution for the confidences of the data (not true, currently very left-skewed) and samples using a normal probability distribution function until a certain quota is met (currently )
  * Hit rate: 0.873 for normal, 0.821 for IQR
    * Goal is to optimize hard cases while maintaining a high hit rate
  * Spread is generally between 0.9 and 0.999 for IQR, between 0.75 and 1.00 for Normal
    * Samples are still very left-skewed - unsure if there's a way to resolve this
* Researched literature on existing unsupervised and active learning methods for retraining
  * Shouldn't have large changes to baseline network weights when retraining
  * Look into Expectation-Maximization algorithms to examine the probability space without knowing the ground truth
  * Should I make a test set for the KAIST data?
  * It's reasonable that performance improves on KAIST data, but not on 74K data, from results obtained by Bruzzone and Prieto on remote sensing data
    * They used a Maximum Likelihood Classifier rather than an object detector - should train something similar (expanding beyond letters)
  * Equal amounts of training data for baseline and retraining should be good - augment with goal of 10K images per class

**Tuesday, July 14**

* Attended daily scrum meeting
* Read about hyperparameter optimization, batch size vs. learning rate, and gradient optimization
* Investigated results of normal curve
  * Model improvement decreased to around 0.80 average precision
  * Likely due to lowered hit rate and easy overfitting of current model parameters
* Modified precision/accuracy metrics in time series to be the mean prec/acc, not the *overall* prec/acc
  * Shifts results slightly, but not significantly
  * Same trends for median thresh sampling are evident
* Began retraining baseline character data based on modified parameters and augmentation techniques to determine changes/improvments
  * Using full data set to train/test, not undersampling
  * Batch size of 32, 15K images per class
* Beginning to rewrite image augmentation pipeline to account for bounding box transforms