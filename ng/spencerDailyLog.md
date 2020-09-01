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

**Wednesday, July 15**

* Attended seminar on final presentations
* Analyzed results from retrained character model
  * Improved mAP from a baseline of 0.86 to 0.90 on test data
* Ran median threshold retraining for 25 epochs, using 10K images per class and batch size of 16
  * Similar results, with a slight peak in mAP for KAIST data (1.7% increase) then hovering around original value
  * mAP for test data decreased again
* Running tests on normal and IQR samples to determine if increased hit rates (89.95% IQR, 90.83% normal) have an impact while preserving "hard" cases
* Rewriting albumentations source code to account for elastic transform bounding boxes

**Thursday, July 16**

* Attended scrum meeting
* Attended student seminar on scientific visualization
* Benchmarked/analyzed results from IQR and Normal distribution sampling/retraining run overnight
* Restarted median threshold and IQR retraining after starting from the wrong checkpoints (with less accurate baseline model)
* Worked on presentation and compiling graphs/data
* Attended student cohort session
* Continued working on presentation and documenting code changes

**Friday, July 17**

* Compiled and analyzed results from revised median threshold retraining
  * 2.2% increase at the peak, lot more object misses
  * Later epochs have more object detections, which could be confounding the results
  * High hit rate appears to be a major influencing factor
* Revised benchmarking and visualization tools for greater modularity
* Gave presentation on results of initial retraining methods and results
* Read paper on uncertainty estimation (Geifman et. al)
  * Describes estimation method of uncertainty ("negative confidence") based on confidence scores over time and a given classifier
  * Early stopping algorithm finds a particular confidence function that leads to a best fit
  * Simple approximation of the algorithm could be implemented in my model instead of current confidence
  * Confidences tend to "overfit" as training goes on (shown in my results)

## Week 6

**Monday, July 20**

* Finished bounding box transformation feature alongside augmentation
  * Previously, bounding boxes were assumed to be the entire augmented image
  * Relies on my custom [albumentations fork](https://github.com/spencerng/albumentations)
  * Added support for optical and elastic deformations
* Should change number of misses to recall (TP / (TP + FN)) in analysis
* Beginning further review on uncertainty estimation methods
  * Many manipulate confidences over time, as a cumulation of scores as training goes on
  * Need to further look into this
* Implemented the average early stop methodology based on existing benchmarking script
  * Averaged confidence scores for a particular image from epoch 0 to 74, with an interval of 3
  * Took the most confident class at the end, making precision a "binary" choice with a cutoff confidence of 0.5
  * Cutoff may need to change? Average confidence score is now around 0.75
* Has a much more normal distribution of confidences now
  * General precision vs. confidence curve shows positive trend again
  * Misses are more right-skewed
* Re-ran sampling methods
  * We now find the quartile and Normal PDF metrics on the set of confidences scores that are individually greater than 0.5
    * Might frequently sample images with a maximum confidence below 0.5 otherwise
  * Median determination remains the same, as median conf is always above 0.5
  * Recall is 100% now, as we don't "snapshot" our progress
  * p = 0.4 for Normal distribution, get fewer samples and higher hit rate that way
    * Still enables us to sample values with confidence below 0.5
* Overall hit rate increases
  * 99.25% for median
  * 97.77% for quartile
  * 96.37% for normal
* Now retraining with same parameters (25 epochs, batch size = 16, 10K images per class)
  * Possible source of error/confounding variable could be modified augmentation with more precise bounding boxes
  * Elastic transforms sometimes yield incorrect bounding boxes
  * Loss function appears to have high values

**Tuesday, July 21**

* Attended AI/ML scrum meeting
* Analyzed results from the three retrained models (third iteration of these sampling techniques)

Method | Hit Rate | Peak mAP (sample) | Epoch of Peak | Epoch 99 Recall
------ | -------- | -------- | ------------ | -------
median thresh | 99.25% | 0.907 | 92 | 99.9%
quartile range | 97.77% | 0.902 | 88 | 98.7%
normal pdf | 96.37% | 0.899 | 96 | 99.0%

* Overall trends
  * Recall improves or stays the same for all methods
  * Peak mAP is a slight improvement over the baseline of 0.888
  * Appears to be a correlation between peak mAP and hit rate
  * Test set accuracy decreases again over time but tends to stablize as training progressed
* Now running tests for 25 more epochs - to see if results converge or continue to improve
* Attended seminar on using neural nets for modeling particle accelerators
* Evaluated retrained models using the rolling confidence average method (starting at epoch 0, end at 99, delta of 3 epochs) as a second measure of improvment
  * Precision can be measured here too, using a cutoff score of 0.5
  * Benchmark confidences had an average of 96.5% precision, 87.3% accuracy by ground truth classes
  * The mean precision across most confidence scores above 0.5 is now nearly 100% for all three methods

Method | Avg. Accuracy | Avg. Precision
------ | ------------- | -------------
median thresh | 89.5% | 97.5%
quartile range | 89.0% | 97.1%
normal pdf | 88.3% | 96.7%

* As somewhat expected, benchmarking with just the retrained epochs (75 to 99) leads to higher accuracy (due to fewer false negatives) but lower precision (due to more false positives)
  * Normal distribution still created for the misses, left-skewed distribution for the hits

* Reading papers about early stopping without a test set/ground truth
  * Could be used for iterative retraining/sampling
  * Should we create a test set from the KAIST data to show improvements? Slightly concerned about "overfitting" on the entire sample pool
  * Could measure the size of the mini-batch gradients over time
  * Stop when it is unlikely that it deviates significantly from the "true" error of the population
    * Note that even the training set is considered a subset of the population
  * Look at section 2.4
* To do
  * Implement gradient descent early stop based on paper
  * Develop pipeline to continuously retrain on a sample
    * Monitor for overfitting?
  * Evaluate convergence or growth results for the 25 extra epochs of retraining (quartile range and median thresh) on lambda

**Wednesday, July 22**

* Attended student seminar on deliverables and sample student presentations
* Analyzed results for the additional 25 epochs
  * Results seem to converge near the maximum, with some peak values slightly (not significant) above the first 25 epochs of retraining
  * Slight increases in precision for all cases on the sample set
  * Test set decreases over time
* Implementing gradient descent early stop
  * Need to compute variance of the individual batch gradients and the batch gradients themselves
  * Looking at using the `backpack` library to do so
  * Facing some errors in the module's shape with recursive extension through the model's layers
  * May need to rewrite part of the backpack module myself

**Thursday, July 23**

* Attended ML scrum meeting
* Discussed with Sean about implementing mini-batch variance in YOLOv3-Pytorch
* Implemented early stop criteria via `backpack`
  * Rewrote part of the library (forked on my [GitHub](https://github.com/spencerng/backpack)) to ignore `None` types (when evaluating) and scalars (for loss functions)
  * Only convolutional layers are accounted for now
  * Ran into issue with nan values due to variance of certain samples being 0
    * Ignoring those layers when calculating the overall loss criteria
* Ran test of loss criteria on retraining examples
  * Difficult to determine if it works, as Lambda is under high load
  * Values of around -9 when retraining on epoch 74, median threshold
  * Expected behavior, as loss is high early on and my batch size is reduced (ran out of VRAM on batch size of 16)
* Rewrote retraining pipeline to incorporate early stopping
* Looking into how significantly YOLO's loss function affects its gradients
  * Would inaccurate/uncertainty in ground truth affect loss and the mini-batch gradients to the point where we will never meet the criteria to stop?
  * May need to make a custom loss function and/or weight samples somehow otherwise

**Friday, July 24**

* Checked on progress from overnight training
  * Stop criteria still hovers around -9.0, at epoch 90
  * This is approaching the peak of the precision in the previous retraining
  * Continuing problem: how do we detect the peak of training precision wrt ground truth?
  * At the same time, we aren't 100% certain of the ground truth
* Looked at parallelized retraining to take advantage of lambda's multiple GPUs
* Possible solutions
  * Custom loss function weighing loss on more confident images (using modified confidence function) more heavily
    * Evidence shows that precision and confidence is positively correlated
    * Simple (normalized) proportion of YOLOv3's loss
    * This should improve overall precision as well
  * Re-evaluate stop criteria
    * Simple threshold shift
    * See if there's a (slight) trend as retraining goes on
    * Use smaller alpha level for exponential smoothing to reduce noise and better see trends
  * Create a validation set from the samples, disjoint from the training set
    * Use a certain proportion of the most confident (top 10%?) samples
* Attended student physics seminar on research at the South Pole
* Reading papers on how effects of uncertainty in training set/ground truth can be minimized while training
* Saw AI/ML presentations
* Discussed with Sean and Nicola about moving forward and clarified overall project goals
  * Autoencoding may be a way to separate hard samples
  * Unsure about early stopping
  * Bandwidth problems are both literal and with annotating data for the ground truth
  * Generalizing to an environment may be a beneficial aspect
* Documented the revised pipeline in preparation for refactoring code and testing more sampling methods next week
  * Idea is to create more modular functionality, in preparation for deployment and testing on other models
* Running experiment on lambda to determine effectiveness of gradient early stop criteria without validation when ground truth is known

## Week 7

**Monday, July 27**

* Begin refactoring code for entire training, sampling, and retraining pipeline
* Attended virtual tour of the ALCF
* Generalized iterative stratification sampling algorithm to include a user-defined number of subsets
* Working on incoporating augmentation into the pipeline

**Tuesday, July 28**

* Attended scrum meeting, learned about Dockerizing code
  * Better to simulate the pipeline, rather than using virtual Waggle
* Continue refactoring code
  * Wrote functions to resample while retaining a proportion of the old training data
  * Need to fix to account for the same train/valid/test splits
* Need to test retraining functionality and clean up old code

**Wednesday, July 29**

* Created functions to extract a proportion of the train/test/validation set for previously-seen images
* Implemented new early stop methodology
* Deployed retraining model overnight, with the same sampling methods
* Fixed bug in creating splits for the bandwidth problem

**Thursday, July 30**

* Attended AI/ML scrum
* Created functions to save "reload" splits for a given image set in the process, as skipping augmentation means `random` is no longer seeded properly
* Analyzed overnight results, with an early stop at epoch 46 for the initial training
  * Program had a bug after first iteration of median threshold sampling, need to fix and re-run
* Attended Learning Off the Lawn presentations
* Worked on presentation
* To do: rewrite some benchmarking tools to account for testing checkpoints on one iteration's test sets in addition to all iterations
  * Need to compile split list files into one image folder

**Friday, July 31**

* Completed loss and mAP benchmarking over epochs on a validation set
* New train/test set stops after epoch 48, using that as the baseline
* Early stop and iterative retraining appears to be working, need to write pipeline to analyze results based on sample set files
* Gave presentation to the SAGE team on my revised pipeline
  * Rick suggests using Monte Carlo dropout for an improved measurement of confidence
    * Seems similar to the smoothed average of model output confidences, except we bias more towards the current epoch model
    * Worth looking into in the future
  * Sean suggests a continuous pipeline for analyzing, in addition to retraining
    * Will be building in the near future, using only the names/prefixes of the training methods
* Outline methods for benchmarking test sets from various file splits

## Week 8

**Monday, August 3**

* Load results from retraining trials overnight
  * While training, removed the incomplete last batch due to near-zero validation/test set
  * Not a situation that would occur in real life
* Fixed bugs with loading lists
* Basic retraining parameters

Method | Sample Size | Total (Re)Train Set Size | Validation Set Size | Test Set Size | Stopping Epochs
------- | -------- | --------- | --------- | ------- | -------
(initial) | 4351 | 3048 | 651 | 652 | 48
median-thresh | 253 | 272 | 33 | 32 | 54, 60, 66, 72, 78, 102, 114
median-below-thresh | 254 | 273 | 33 | 32 | 48, 56, 70, 76, 82, 88, 94, 100
normal | 300 | 325 | 37 | 37 | 54, 60, 70, 76, 82, 98, 112
iqr | 250 | 269 | 32 | 32 | 54, 60, 74, 82, 88, 94, 108

* Preliminary Notes
  * Due to the 80/10/10 split for retraining and the lack of class balancing (to account for real life situations) when creating batches for sampling, some classes were only represented once or twice in the iteration test/validation sets
  * The iteration rate limit of 500 images and the sampling methods used (using ~50% of the sample batch) makes it unlikely that we will hit our bandwidth limit of 300 images
  * Due to variations in the iterative stratified sampling algorithm, counts for train/validation/test sets may be off by 1-2 images
  * The normal sampling method had *p*=0.4 for values within one standard deviation of the mean confidence. This should probably be increased (to 0.75) and re-run (now updated!)
  * The old quartile range (0.25 to 0.5) was replaced with an interquartile range this time for a greater sample size
  * The median-below-thresh method was the only one to hit an early stop with the minimum number of epochs (7 batches * 3 strips * 2 epochs/strip = 42)
  * This also occurred in the original run of the normal method, when *p*=0.5
    * In the re-run, the last iteration was the only one to break this trend
* Analyzed trends between the precision/accuracy of the sample at the edge in relation to retraining time
  * Note that ground truth is corrected before we retrain images
  * These are the *entire* samples (not just test sets) before we retrain on that batch

**Median Threshold**

```
   Batch  Avg. Prec  Avg. Acc  Epochs Trained
0    0.0   0.938086  0.855955             6.0
1    1.0   0.973196  0.894340             6.0
2    2.0   0.964517  0.874532             6.0
3    3.0   0.982543  0.911128             6.0
4    4.0   0.981681  0.886946             6.0
5    5.0   0.956378  0.856634            24.0
6    6.0   0.983182  0.904097            12.0
```

**Below Median Threshold**

```
   Batch  Avg. Prec  Avg. Acc  Epochs Trained
0    0.0   0.938086  0.855955             8.0
1    1.0   0.987797  0.906188            14.0
2    2.0   0.971922  0.879432             6.0
3    3.0   0.972278  0.924664             6.0
4    4.0   0.983442  0.925842             6.0
5    5.0   0.971272  0.852532             6.0
6    6.0   0.990739  0.883677             6.0
```

**Normal Sampling (revised for *p*=0.75)**

```
   Batch  Avg. Prec  Avg. Acc  Epochs Trained
0    0.0   0.938086  0.855955             6.0
1    1.0   0.982737  0.902626             6.0
2    2.0   0.975404  0.892362            10.0
3    3.0   0.979842  0.917734             6.0
4    4.0   0.986752  0.895814             6.0
5    5.0   0.960903  0.881411            16.0
6    6.0   0.984296  0.900177            14.0
```

**IQR Threshold**

```
   Batch  Avg. Prec  Avg. Acc  Epochs Trained
0    0.0   0.938086  0.855955             6.0
1    1.0   0.986422  0.898950             6.0
2    2.0   0.965950  0.893017            14.0
3    3.0   0.966728  0.910968             8.0
4    4.0   0.980250  0.912477             6.0
5    5.0   0.970522  0.879621             6.0
6    6.0   0.982247  0.897518            14.0
```

* There aren't any clear trends, except precision/accuracy both noticeably increase after the first retraining batch
* It seems logical to have several test sets for analyzing results over time
  * Test sets filtered for only sampled (KAIST) images
  * Initial (74K) test set
  * Data from all test sets
  * Only data from the (combined) iteration test sets (75% KAIST + 25% seen data)
  * Test set for the current iteration (creating several disjoint segments)
* Wrote scripts to benchmark and visualize results from IQR and median threshold
  * It appears as though batches beginning with low (< 0.85) precision *do* increase slightly over the course of the retraining iteration
    * Interesting due to the fact that training and testing data aren't related, aside from class balance (and even that is offset by augmentation)
  * Some batches start with high precision (> 0.95) due to small sample size
  * Initial testing set appears to always decrease
  * Testing set from the sample slightly increases
  * Unsure if the additional 25% of data retained is useful
* Running a model with samples below the median threshold overnight

**Tuesday, August 4**

* Attended AI/ML scrum
* Analyzed results for the revised normal sampling distribution and below median threshold sampling (updated above)
* Same general trends show with some individual iterations improving, though others have a downward trend
  * Might be confirmation bias, re-running analyses with the average confidence method might help smooth these trends and eliminate some noise
* Realized that the individual images in the batch splits are the main control variable between the different sampling methods
  * Some sampling methods are pseudo-random (e.g. Normal distribution), as is creating test splits and enforcing the bandwidth limit
  * Consequently, there's not good baseline, especially with a small sample size on iteration test sets
    * Some might be filled with "harder" images intrinsically
    * We also self-select for harder/easier images by the sampling method themselves
  * Batch splits should be fair, though aren't completely balanced for class (only issue)
* Here are some results that might establish a trend (measured with confidence based on the average of 10 epochs):

**Avg. Precision by Batch Iteration**

```
Batch  median-below-thresh  median-thresh    normal       iqr
0                 0.938086       0.938086  0.938086  0.938086
1                 0.987797       0.973196  0.982737  0.986422
2                 0.971922       0.964517  0.975404  0.965950
3                 0.972278       0.982543  0.979842  0.966728
4                 0.983442       0.981681  0.986752  0.980250
5                 0.971272       0.956378  0.960903  0.970522
6                 0.990739       0.983182  0.984296  0.982247
```

**Avg. Accuracy by Batch**

```
   median-below-thresh  median-thresh    normal       iqr
0             0.855955       0.855955  0.855955  0.855955
1             0.906188       0.894340  0.902626  0.898950
2             0.879432       0.874532  0.892362  0.893017
3             0.924664       0.911128  0.917734  0.910968
4             0.925842       0.886946  0.895814  0.912477
5             0.852532       0.856634  0.881411  0.879621
6             0.883677       0.904097  0.900177  0.897518
```

(Everything generally increases, and less significant gains here may be due to a good number of false negatives still)

**Avg. Recall**

```
0             0.900911       0.900911  0.900911  0.900911
1             0.905850       0.907530  0.909032  0.902838
2             0.889368       0.888573  0.900356  0.908902
3             0.943451       0.914940  0.925233  0.929093
4             0.931831       0.890656  0.896577  0.920041
5             0.858231       0.874567  0.899269  0.888709
6             0.878744       0.904750  0.901845  0.902121
```

**Mean Average Confidence (of Batch)**

```
   median-below-thresh  median-thresh    normal       iqr
0             0.770385       0.770385  0.770385  0.770385
1             0.745363       0.768198  0.766087  0.766286
2             0.743698       0.750652  0.755300  0.754191
3             0.763418       0.753936  0.759225  0.769329
4             0.750285       0.756568  0.741312  0.759916
5             0.731022       0.736236  0.735920  0.719376
6             0.736928       0.747115  0.736947  0.754437
```

(There's a general downward trend here, which could suggest the model being more generalized, but it's hard to draw real conclusions)

* Although this isn't controlled and isn't a "result," it might be interesting to analyze the original precision/accuracy of the images sampled that are used for retraining/testing. This might influence our results later on:

**Avg. Precision of Sample**

```
   median-below-thresh  median-thresh    normal       iqr
0             0.870883       0.969004  0.928215  0.940445
1             0.967234       1.000000  0.983816  0.995328
2             0.941667       0.986251  0.974888  0.976197
3             0.942708       1.000000  0.974288  0.981788
4             0.960960       1.000000  0.982324  0.974700
5             0.954915       0.979197  0.963586  0.963333
6             0.973485       0.997024  0.993584  0.975148
```

**Avg. Accuracy of Sample**

```
   median-below-thresh  median-thresh    normal       iqr
0             0.715664       0.918227  0.887453  0.940445
1             0.816597       1.000000  0.943243  0.995328
2             0.771624       0.986251  0.918670  0.976197
3             0.853971       1.000000  0.934677  0.974246
4             0.844122       1.000000  0.932486  0.965441
5             0.725523       0.979197  0.914424  0.948592
6             0.771347       0.997024  0.937082  0.962328
```

This seems to confirm the large number of false negatives, as does the decreasing recall

**Avg. Recall**

```
   median-below-thresh  median-thresh    normal       iqr
0             0.773832       0.946002  0.948857  1.000000
1             0.794570       1.000000  0.955263  1.000000
2             0.754249       1.000000  0.937024  1.000000
3             0.873358       1.000000  0.953326  0.992424
4             0.837966       1.000000  0.943937  0.990741
5             0.700562       1.000000  0.944194  0.982077
6             0.727435       1.000000  0.940697  0.986111
```

**Mean Avg. Confidence (of Sample)**

```
   median-below-thresh  median-thresh    normal       iqr
0             0.644174       0.824698  0.800676  0.825368
1             0.609246       0.890358  0.799843  0.825076
2             0.602483       0.892017  0.806026  0.815630
3             0.639064       0.888963  0.789031  0.829778
4             0.611176       0.900145  0.781610  0.812777
5             0.591291       0.881563  0.781250  0.766124
6             0.588583       0.893725  0.785986  0.824303
```

* Power went out; no longer analyzing results as they are stored on Lambda
* Rewrote confidence averaging to perform non-max suppression instead
  * Not a true average now, but should generalize better for other models
  * Labels now based on model output bounding boxes as well, not hard coded to be the entire image
  * **To do**: actually test this thing once I have the (processing) power to do so
  * **To do**: make a way to have multiple labels per image, perhaps removing reliance on spreadsheet
  * Need to account for video inputs too
* Starting to write article on my project for the Sage website

**Wednesday, August 5**

* Wrote very rough draft of the article when there was a power outage
* After power restored, tried to debug code for NMS averaging, unsuccessful
* Deployed simple sampling method with strict cutoffs above/below 0.5 confidence

**Thursday, August 6**

* Attended AI/ML scrum
* Reran strict cutoff retraining after Pytorch was out of memory on Lambda, causing a crash
  * During training, the lower cutoff (below 0.5) seems to eliminate classes over time, as some have very high confidence scores
* Revisited NMS averaging by creating an algorithm that takes overlapping bounding box areas and averages them
  * This is slightly different from NMS, as we don't strictly eliminate boxes, but we still rely on IOU to determine overlap
  * Class confidences are summed for each bounding box, then divided by the number of epochs
  * Best bounding box (highest avg. confidence) in each region is retained
  * Sampling only keeps the class label and not the bounding box generated by YOLO, as we assume bounding boxes are manually corrected
  * Confidences of isses are slightly more left skewed than before (hard to say due to small sample size), but are still differentiable wrt hits
* This algo will allow us to sample on things like car detection, with multiple ground truth labels per image
  * Goal for next week, as work with characters is essentially done, except for binning evenly
* Also need to analyze (averaged and non-averaged) results for strict cutoffs tomorrow

**Friday, August 7**

* Checked progress on strict cutoff retraining
  * Still ongoing, last iteration seems to never hit the stop criteria
  * A sign that we have constructed a poor validation set?
* Rewrote part of the analysis script to easily tabulate data via the command line
  * Can cross-compare (mean) precision, accuracy, recall, and confidence, filtered for samples as an option
  * We might be able to correlate sample accuracy/precision and subsequent improvements
* Documented config usage
  * Still need to document implementing sampling and custom models
* Attended demo presentations
* Beginning work on binning algorithm for sampling
* Also working on matching multi-region detections for hits/misses
  * Need a way to pair up detections and ground truth
* Deployed training using a Normal PDF centered at 0.5, with standard deviation of 0.25
* Aside from this, a binning with undersampling and even distribution across confidence quintiles will likely be the last method I try
* Might move onto testing these methods on preexisting models

## Week 9

**Monday, August 10**

* Wrote rolling average functions for benchmarking
* Wrote analyses functions for correlation between two variables (e.g. sample confidence and resulting precision on next batch iteration)
* Finished multilabel benchmarking options
* Created confidence visualization function of sampled images and those used for retraining
* Finished binning sampling algorithm
* Added support for non-stratified sampling
* Ran rolling average analysis on previous sampling methods
  * Todo: look into other methods for judging sample batch precision/accuracy, aside from the "built-in" (already generated) average of 10 checkpoints
* Running binning of quintiles and normal curve overnight

**Tuesday, August 11**

* Attended AI/ML scrum meeting
* Converted confidence metric to account for object score in addition to confidence score
* Benchmarked initial checkpoints against all test sets for a fair standard of comparison
  * Oddly, performance on the first batch set is worse than others (0.938 vs. ~0.96 precision)
  * It's possible we randomly generated a "harder" set, as the checkpoints used in this case don't change
* Added feature to benchmark sampling methods on batch sets via a rolling average instead of the linearly-spaced average
  * Using 5 epochs, this led to far worse results when looking for improvement gains
  * Likely going to stick with the linear average - an option when running inference under real-world conditions
* Some correlation trends, particularly in confidence of sample vs. accuracy in next iteration
  * Need further analysis - do we use the outlier of median-below-thresh?
* Cleaned up maingate dataset, in preparation for model training
  * Might use a simple object detector for cars, as we don't have enough labels for each class for retraining

**Wednesday, August 12**

* Investigated Stanford car dataset again
  * We could use different types of cars (e.g. SUV, Minivan, Sudan) as classes, which would give us enough data to work with per class
  * However, the maingate dataset doesn't have these labels, only the make and model
* Deployed KAIST retraining models with parallelization to verify results of the first set of experiments
  * This uses the new confidence score metric, accounting for object scores and multiple labels per image
  * Should have complete results by tomorrow morning
* Refactored code for analysis tools, adding to enable/disable lines on the timeseries graph
  * **TODO**: combine the current iteration lines into one
* Revised website article in preparation for publishing onto Wordpress
* Added random sampling method and deployed it
  * Should consider running some sampling methods without stratification, such as median-below-threshold
* Continued documenting usage of sampling and analysis tools

**Thursday, August 13**

* Attended AI/ML scrum
* Cleaned up data from overnight retraining and fixed errors
* Benchmarked against various sampling methods in the second run
* Wrote method to determine variance/standard deviation of object and class confidence scores when computing average
* Combined current iteration lines into one series
* Modified comparison method to include the change in metrics (e.g. accuracy/precision) against the baseline model
* Begin documenting results in presentation
  * After lowering threshold for a positive result to 0.3, we see a greater correlation in mean sample conf. vs. precision/accuracy in next iteration's batch set inferences
  * This also fixes the lowered accuracy seen before
* Researched potential datasets
  * Dogs is another possibility for object detection
  * Google's Waymo dataset has many vehicles, but no granular data like make/model
  * Need to continue looking into this

**Friday, August 14**

* Ran regression and time series tests after benchmarks were running overnight
  * Sample conf vs. next batch precision yields the best correlation, even compared to sample precision
* Documented results in presentation for next week
* Briefly revised analyses tools to incorporate a matplotlib graph of confidences, alongside the generated PDF
* Attended AI/ML presentations
* Wrote script to label Stanford car dataset, sorting by car types (e.g. sedan, SUV, cab)
* Deployed model on Lambda to train for the weekend
* Writing tools to help manually annotate the maingate dataset, hopefully each class yields similar types
  * Otherwise, may search the internet for similar images

## Week 10

**Monday, August 17**

* Tried to troubleshoot why car type model deployed overnight failed
  * CUDA shapes are wrong
  * Bounding boxes are transformed outside of the image
* Fixed bugs in albumentations pipeline
  * Added BboxParams object for minimum visibility
  * Partially rewrote library to remove rounding errors
* Deployed model again, targeting 10k images per class
  * Augmentations appear to be sound
  * After low mAP with 30 epochs of training, increased to 15k images per class
  * SUV, Van, and Cab classes appear to have highest precision
* Researched similar datasets online
  * Stanford dataset generally used for classification, sometimes with low mAP
  * COCO could be used for training, in an instance of a binary classifier (vehicle and non-vehicle)
    * Essentially judges improvement in object detection within maingate images
  * We could also crop images of the maingate cars, in order to help with object detection
  * Lowering the number of classes in the model is a third option
* Wrote tool to help manually label classes (vehicle type) for the maingate dataset
  * Began labelling, though we skew towards high amounts of SUVs and sedans
  * Not sure if the data is very usable for retraining

**Tuesday, August 18**

* Attended AI/ML scrum meeting
* Fixed bug with image resizing in inference - cause of low precision before
* Performance on validation set:

```
Early stop at epoch 104:
+-------+-------------+---------+
| Index | Class name  | AP      |
+-------+-------------+---------+
| 0     | Cab         | 0.68267 |
| 1     | Convertible | 0.44596 |
| 2     | Coupe       | 0.35592 |
| 3     | Hatchback   | 0.14437 |
| 4     | Minivan     | 0.29072 |
| 5     | SUV         | 0.59701 |
| 6     | Sedan       | 0.42085 |
| 7     | Van         | 0.69161 |
| 8     | Wagon       | 0.19666 |
+-------+-------------+---------+
---- mAP 0.42508408587691826
Previous loss: 650.4346039295197
Current loss: 651.8103536367416
```

* Good baseline, except running inference roughly on the maingate dataset leads to many false positives
  * Environmental features detected as cars
  * Haven't validated against manual ground truth labels yet, except cars do appear to be detected
* Running with the average method on the combined initial Stanford set leads to an accuracy of 0.729 and precision of 0.714, with 468 images
* Need to figure out if I'll proceed with this or somehow alter classes to make the problem easier
  * Right now, false positives would drive down accuracy and precision significantly - need to benchmark a running average on the sample set to see
* Reading papers about false positives within object detection
* Wrote code to parallelize training pipeline
* Documented more about sampling methods

**Wednesday, August 19**

* Looked into hard false positives in the maingate dataset when running the Stanford model
  * Separating the classifier with the detector (and doing more complex analysis with the object confidence scores) may work, but the goal is to create a generalized pipeline with YOLO
* Fixed bugs in classifying false negatives
  * Verified results on old character model; few changes in actual trends/results
* Sped up augmentation increase calculation and sampling methods by optimizing algorithms
* Documented sampling methods (and creating custom ones)
* Finished labeling maingate set by car type
  * Dataset contains 1500 images, 1800 labels

**Thursday, August 20**

* Attended AI/ML scrum meeting
* Looked into NVIDIA car data sets
  * Track 1 (video stream) has no ground truth, as the goal is counting vehicles in each frame
  * Track 2 has ground truth labels for truck/car and isn't in context
* Found VeRi dataset also on LCRC
  * Has car types and many images but no images in context (pre-cropped)
  * GitHub link seems to indicate full scene images (from a camera feed) are available, I've reached out to the authors
* Other car datasets either aren't labeled with the data we want, are from the wrong perspectives,and/or aren't in context
* Sticking with the VeRi dataset in lieu of the Waggle one for sampling due to its more reliable labels
  * Unfortunately, this won't be a true object detection problem again for now, but could be if I get the full dataset
    * We still evaluate using object detection methods
  * Classifying based on one of five vehicle types

Class     | Stanford | VeRi
----------|----------|-------
Cab/Wagon | 1430     | 3094
Hatchback | 1103     | 2769 
Sedan     | 3787     | 25011
SUV       | 2855     | 6617
Van       | 660      | 2323
Total     | 9835     | 39814

* To balance between consistency and the KAIST experiment, where the training set size was around 8% of the original training set, 3000 images per batch will be used, with an 1800 image bandwidth limit and a 70/15/15 split
  * 75% new data will once again be the new iteration set size amount, with seen images added in for the other 25%
  * This will lead to 13 full batches
  * Hopefully, this is also a more realistic simulation of a video feed
  * Other parameters remain the same from the KAIST run, though positive threshold may change if confidence distribution is significantly different
* Wrote code to properly parallelize running multiple sampling/retraining methods
* Running new model/classes overnight, starting from intial training

**Friday, August 21**

* Finished training initial model at epoch 154:

```
+-------+------------+---------+
| Index | Class name | AP      |
+-------+------------+---------+
| 0     | cab        | 0.71968 |
| 1     | hatchback  | 0.32797 |
| 2     | sedan      | 0.78136 |
| 3     | suv        | 0.73037 |
| 4     | van        | 0.77173 |
+-------+------------+---------+
---- mAP 0.6662235857615968
Previous loss: 344.7222201228142
Current loss: 347.083483338356
```

* Fixed bugs in GPU multiprocessing
* Updated presentation with general overview and brief updates on car detection
* Gave presentation
  * Might want to look into measuring improvement as a function of bandwidth limit/sample batch size, to evaluate real-world tradeoffs
  * Could also augment seen images and sample images separately, such that we use as many different raw images as possible for the 75%/25% split
    * i.e. Augment a lot more sample images, fewer seen images
* Fixed multiprocessing bugs

## Week 11

**Monday, August 24**

* Analyzed results from overnight training
  * Below median threshold method drastically improves performance
  * Hard to compare others, as multiprocessing crashed in some subprocesses
* Debugged multiprocessing code
  * Need to relink tensors to the right device
  * May not support non-CUDA devices right now
* Began documenting and linting code in preparation for Dockerizing
* Fixed parallelization bugs, now running all training methods overnight to analyze data
* To do
  * Finish linting/documenting code
  * Write analysis tool usage in README
  * Finish writing Sage website article
  * Analyze results when retraining is complete
  * Write a more technical paper with results
  * Prettify CLI outputs

**Tuesday, August 25**

* Had ML scrum meeting
  * Discussed creating a baseline model with the sample set, adjusting badwidth parameters, putting code into production
* Running various tasks on compute nodes:
  * lambda1
    * Training a baseline model with the sample set (15k images per class for augmentations, like the original) in `~/veri-baseline`
  * lambda2
    * BW limit of 900 (30% instead of 60% of batch size) in `~/car-retrain-bw` - need to benchmark when finished
  * lambda3
    * Test series benchmark (now in parallel) - need to rerun once lambda4 is complete
    * Retraining using only with new `true-random` method (no stratification by inferred class) in `~/stanford-car-cleanup`
  * lambda4
    * Continuing the original retraining in parallel, some sampling methods have finished
* Creating a common test set for the baseline model is an interesting problem
  * Iteratively train with batch sets again? 
  * Created one model with the entire sample set, another with a batch training method
    * Entire sample set yields mAP of 0.755531 on the test set, not a particularly meaningful metric for comparison
* Continued linting/documenting code
* Updated website article based on progress thus far
* To do remains same as yesterday

**Wednesday, August 26**

* Continuing to revise webisite article
* Refactored large portion of the code, condensing long functions
* Analyzing results as they come in on Lambda nodes
  * Significant correlations between epoch training time and next batch precision/accuracy
  * Lowered bandwidth seems to follow similar trends (except in the random case), though increases in performance are lower
  * We now see a *positive* correlation between sample precision and resultant precision
  * Baseline (batch training using all VeRi data) gives around 0.88 mAP for a batch, compared to IQR's peak of 0.85 mAP at the last batch
    * mid-thresh also yields good improvements
* Need to analyze the confidence distributions further
* Started Overleaf template with Bibtex for tracking references

**Thursday, August 27**

* Attended scrum meeting
* At the suggestion of Sean/Nicola, created new test set from the 3k image pool of batches 11-12, plus the leftover (~900) images from batch 13
  * This means we can only benchmark up to batch 10, but this should provide sufficient results
* Modified analysis tool to allow for this, for any given number of complete batches
  * Need to refactor code to make it a bit less hackish
* Continuing parameter sweep, with 15% and 45% bandwidth limits
* Continued writing article on website

**Friday, August 28**

* Examined epoch benchmarks

VeRi baseline (sample 100%; rolling avg. of 10 epochs):

```
   batch      prec       acc      conf  conf_std    recall  epochs_trained
0    0.0  0.809589  0.755719  0.674711  0.431388  0.908734            30.0
1    1.0  0.852310  0.788701  0.657602  0.409036  0.899350             8.0
2    2.0  0.854484  0.793635  0.668532  0.416112  0.909814            18.0
3    3.0  0.854563  0.801074  0.692672  0.413303  0.919199            14.0
4    4.0  0.840208  0.794706  0.675294  0.411692  0.926882             8.0
5    5.0  0.859341  0.802571  0.668046  0.408447  0.915760            12.0
6    6.0  0.870879  0.814936  0.681441  0.415694  0.918452            10.0
7    7.0  0.868996  0.812889  0.681499  0.412335  0.916710            28.0
8    8.0  0.866795  0.812255  0.690918  0.406877  0.920445            12.0
9    9.0  0.854600  0.796927  0.675109  0.411986  0.909902           120.0
```

VeRi baseline, linear average:

```
   batch      prec       acc      conf  conf_std    recall  epochs_trained
0    0.0  0.803675  0.715160  0.515056  0.472486  0.809336            30.0
1    1.0  0.809075  0.719986  0.554214  0.468466  0.834157             8.0
2    2.0  0.834892  0.742666  0.574752  0.460996  0.846629            18.0
3    3.0  0.859589  0.769147  0.581723  0.462675  0.859838            14.0
4    4.0  0.857281  0.754984  0.574996  0.459691  0.842253             8.0
5    5.0  0.853002  0.738999  0.539765  0.459104  0.819639            12.0
6    6.0  0.886881  0.795085  0.596742  0.462855  0.871441            10.0
7    7.0  0.846512  0.757801  0.589958  0.461536  0.861282            28.0
8    8.0  0.883353  0.778890  0.581738  0.459638  0.851829            12.0
9    9.0  0.871066  0.768009  0.565899  0.451512  0.846519           120.0
```

* Finished writing website article
* Modifying pipeline for more flexibility with the universal test set method, including the option for linearly-spaced and instantaneous averaging for test set
  * Benchmarking with the test set over the weekend
* Started an Overleaf version of the paper
* Continued documenting code

## Week 12

* Restructured files to train from a single folder with software
  * Modified software to allow for inheriting options from other configuration files
* Continued documenting code and fixing parallelization bugs
  * Beginning documentation on analysis module in separate readme
* Running parameter sweep for 75% bandwidth on lambda4
* Benchmarking and compiling other results:

60% (original set of models), rolling avg:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.321686  0.321686  0.321686  0.321686  0.321686  0.321686  0.321686  0.321686  0.321686  0.321686  0.321686
median-below-thresh  0.320223  0.566349  0.697827  0.758491  0.755854  0.765714  0.789185  0.821933  0.810455  0.835269  0.822061
median-thresh        0.313673  0.616531  0.652751  0.715365  0.736016  0.724016  0.719298  0.769186  0.740376  0.776998  0.772877
normal               0.317570  0.572333  0.666275  0.711811  0.715017  0.742705  0.760688  0.772818  0.772486  0.780649  0.782772
iqr                  0.315524  0.654859  0.707051  0.751441  0.774560  0.795091  0.795974  0.832666  0.811420  0.823101  0.814053
mid-thresh           0.325412  0.534994  0.697393  0.697050  0.691101  0.731962  0.731888  0.786185  0.784851  0.804039  0.813647
mid-below-thresh     0.324730  0.459519  0.584379  0.621256  0.668494  0.728281  0.734114  0.758819  0.756151  0.804033  0.804535
mid-normal           0.314650  0.607750  0.748681  0.751588  0.749616  0.772535  0.771464  0.786568  0.816669  0.802596  0.809535
bin-quintile         0.324346  0.497226  0.676384  0.735730  0.734878  0.766854  0.775464  0.792965  0.796794  0.829504  0.822324
bin-normal           0.324224  0.706249  0.776406  0.723453  0.798009  0.785461  0.786441  0.805086  0.825781  0.870861  0.838447
random               0.324175  0.664610  0.708875  0.772438  0.711601  0.777388  0.810696  0.842694  0.814062  0.837458  0.832281
true-random          0.322322  0.695596  0.760834  0.776015  0.791799  0.802004  0.816370  0.797367  0.840152  0.843444  0.842194
```

60%, linear avg:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.321963  0.321963  0.321963  0.321963  0.321963  0.321963  0.321963  0.321963  0.321963  0.321963  0.321963
median-below-thresh  0.335474  0.353834  0.688226  0.727407  0.729402  0.711172  0.800632  0.801102  0.808303  0.812870  0.814707
median-thresh        0.330740  0.377586  0.424579  0.436776  0.466491  0.496293  0.680336  0.665440  0.717083  0.762175  0.771326
normal               0.332982  0.359267  0.379956  0.471920  0.501548  0.657543  0.690299  0.702534  0.776090  0.797799  0.812589
iqr                  0.337550  0.468219  0.696428  0.779576  0.794571  0.817474  0.786473  0.842919  0.831968  0.840323  0.844784
mid-thresh           0.337429  0.353074  0.574940  0.616649  0.642480  0.759429  0.777875  0.790725  0.772793  0.810419  0.811694
mid-below-thresh     0.327145  0.357638  0.366102  0.390120  0.522075  0.657884  0.672120  0.711622  0.732381  0.734955  0.744962
mid-normal           0.324692  0.359097  0.682694  0.711815  0.713936  0.700590  0.740744  0.758392  0.794134  0.787501  0.787414
bin-quintile         0.332957  0.360785  0.380953  0.611795  0.657772  0.691346  0.750269  0.706997  0.744929  0.739852  0.765043
bin-normal           0.339722  0.526748  0.620610  0.670434  0.743259  0.796030  0.781528  0.792479  0.836589  0.831128  0.816016
random               0.324765  0.435809  0.442236  0.555296  0.613018  0.607872  0.692652  0.715850  0.686260  0.703331  0.740787
true-random          0.332348  0.575560  0.726050  0.772466  0.761177  0.776502  0.749051  0.784855  0.817287  0.814392  0.845290
```

45%, rolling:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.321516  0.321516  0.321516  0.321516  0.321516  0.321516  0.321516  0.321516  0.321516  0.321516  0.321516
median-below-thresh  0.323579  0.623799  0.672230  0.687915  0.708315  0.747885  0.764704  0.787807  0.795450  0.808652  0.820466
median-thresh        0.314335  0.581898  0.638684  0.705034  0.715189  0.723088  0.752405  0.780577  0.777854  0.810732  0.784103
normal               0.308190  0.705405  0.749731  0.769411  0.777046  0.744175  0.792509  0.803290  0.792220  0.827177  0.812009
iqr                  0.320942  0.689809  0.726924  0.758567  0.761780  0.781637  0.764045  0.776288  0.811432  0.803728  0.835755
mid-thresh           0.316522  0.623521  0.695169  0.682175  0.732847  0.753622  0.734086  0.743176  0.777714  0.787981  0.800580
mid-below-thresh     0.319939  0.574458  0.620056  0.712402  0.747578  0.774742  0.815639  0.797142  0.808196  0.812186  0.825136
mid-normal           0.324308  0.615657  0.709181  0.732673  0.764369  0.787254  0.763274  0.806979  0.816991  0.814346  0.797544
bin-quintile         0.324669  0.673980  0.711423  0.747422  0.771610  0.780434  0.784504  0.814774  0.799862  0.814096  0.791991
bin-normal           0.320965  0.708518  0.776506  0.767231  0.785522  0.796193  0.788427  0.808134  0.839861  0.850276  0.842852
random               0.316567  0.713372  0.762899  0.770897  0.794279  0.801353  0.816002  0.814785  0.807181  0.804850  0.833704
true-random          0.313406  0.637437  0.715294  0.743307  0.776609  0.779634  0.774568  0.821980  0.820084  0.817360  0.827541
```

45%, linear:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.332248  0.332248  0.332248  0.332248  0.332248  0.332248  0.332248  0.332248  0.332248  0.332248  0.332248
median-below-thresh  0.333653  0.376840  0.454719  0.438551  0.571123  0.666787  0.693619  0.691160  0.705326  0.751277  0.729723
median-thresh        0.328807  0.373496  0.371694  0.429941  0.482613  0.736676  0.762751  0.797372  0.794474  0.825636  0.795999
normal               0.332834  0.581819  0.687826  0.785173  0.812509  0.802296  0.797742  0.824879  0.788811  0.821718  0.854575
iqr                  0.334545  0.481317  0.567730  0.630863  0.675290  0.706872  0.808201  0.784263  0.814178  0.815213  0.822046
mid-thresh           0.330241  0.365427  0.503030  0.523220  0.514931  0.559604  0.603430  0.754046  0.777134  0.789634  0.808241
mid-below-thresh     0.343303  0.429491  0.434301  0.623103  0.688525  0.721441  0.728872  0.780428  0.813024  0.771598  0.788889
mid-normal           0.335914  0.373082  0.592266  0.612318  0.685774  0.721771  0.716391  0.733993  0.756883  0.770835  0.772850
bin-quintile         0.342066  0.468297  0.502444  0.715224  0.748482  0.755433  0.763073  0.786619  0.777702  0.826080  0.818475
bin-normal           0.326754  0.495206  0.721490  0.764918  0.796639  0.822057  0.819369  0.839815  0.797009  0.826392  0.812993
random               0.345233  0.685426  0.716814  0.734413  0.757115  0.784750  0.776770  0.844303  0.840729  0.862712  0.854391
true-random          0.336316  0.441676  0.524062  0.602626  0.703809  0.728627  0.743918  0.767636  0.811220  0.816632  0.828077
```

30%, rolling:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.313534  0.313534  0.313534  0.313534  0.313534  0.313534  0.313534  0.313534  0.313534  0.313534  0.313534
median-below-thresh  0.318292  0.494182  0.650849  0.673680  0.738656  0.741147  0.760720  0.788911  0.762222  0.793457  0.805585
median-thresh        0.324888  0.685751  0.706387  0.731613  0.780272  0.757901  0.763445  0.788851  0.748505  0.782055  0.774194
normal               0.317295  0.664889  0.700263  0.705660  0.732676  0.752549  0.770377  0.808728  0.775069  0.801078  0.786192
iqr                  0.319972  0.618831  0.686313  0.706684  0.726641  0.761617  0.772618  0.787025  0.785500  0.798974  0.786176
mid-thresh           0.319349  0.587414  0.644386  0.657932  0.726940  0.745610  0.709823  0.749236  0.749438  0.752809  0.758517
mid-below-thresh     0.315307  0.640243  0.734743  0.767340  0.724136  0.758295  0.750563  0.735084  0.773224  0.785140  0.795608
mid-normal           0.324977  0.651337  0.692420  0.722817  0.744587  0.786903  0.792644  0.791339  0.804150  0.812368  0.804165
bin-quintile         0.321601  0.540366  0.696981  0.745049  0.741942  0.747060  0.753647  0.775934  0.764252  0.767166  0.754870
bin-normal           0.319454  0.589726  0.712382  0.698783  0.712644  0.738179  0.712665  0.760823  0.762168  0.783066  0.805034
random               0.316653  0.666529  0.676650  0.745953  0.728218  0.728244  0.808569  0.772265  0.771446  0.778061  0.768822
true-random          0.320800  0.516118  0.660770  0.694179  0.755520  0.733296  0.753954  0.766346  0.777509  0.768974  0.773192
```

30%, linear:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.328340  0.328340  0.328340  0.328340  0.328340  0.328340  0.328340  0.328340  0.328340  0.328340  0.328340
median-below-thresh  0.347105  0.369130  0.387470  0.454256  0.591744  0.601998  0.628188  0.619343  0.643186  0.671458  0.672973
median-thresh        0.338705  0.486667  0.630893  0.684945  0.666825  0.697612  0.660257  0.719664  0.738080  0.742261  0.785707
normal               0.339117  0.413518  0.445461  0.508557  0.660344  0.731049  0.766220  0.757808  0.787682  0.811367  0.770368
iqr                  0.333561  0.379788  0.493235  0.568913  0.650329  0.670839  0.683414  0.784704  0.796502  0.834985  0.820812
mid-thresh           0.328921  0.353972  0.393323  0.430661  0.600166  0.640113  0.697136  0.712609  0.703098  0.719092  0.775062
mid-below-thresh     0.323951  0.542870  0.636513  0.658713  0.652111  0.746216  0.805024  0.792058  0.807068  0.799084  0.792192
mid-normal           0.322859  0.382881  0.512675  0.538323  0.683048  0.761821  0.807748  0.815146  0.819391  0.801109  0.815029
bin-quintile         0.331284  0.366603  0.687844  0.703534  0.739699  0.739638  0.717258  0.756883  0.772302  0.772115  0.800898
bin-normal           0.329700  0.386141  0.522347  0.572576  0.598491  0.570761  0.582520  0.615730  0.711298  0.712148  0.784901
random               0.320247  0.409835  0.506853  0.732923  0.753508  0.784512  0.781634  0.776104  0.790267  0.808247  0.782412
true-random          0.335402  0.367004  0.372648  0.590096  0.641610  0.654868  0.631104  0.678233  0.731899  0.728542  0.726063
```

15%, rolling:

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.324946  0.324946  0.324946  0.324946  0.324946  0.324946  0.324946  0.324946  0.324946  0.324946  0.324946
median-below-thresh  0.310358  0.625131  0.676405  0.640580  0.684277  0.710945  0.677534  0.662412  0.699104  0.709208  0.669829
median-thresh        0.326410  0.643720  0.665397  0.707529  0.751120  0.699299  0.699562  0.737776  0.722257  0.765612  0.706984
normal               0.317753  0.519716  0.708054  0.665022  0.697630  0.719699  0.733486  0.736632  0.678291  0.729307  0.736736
iqr                  0.313078  0.491091  0.637692  0.633401  0.678325  0.675055  0.695530  0.714291  0.650114  0.655665  0.684199
mid-thresh           0.327797  0.638126  0.649869  0.683228  0.730795  0.725325  0.708891  0.717451  0.799856  0.736911  0.751161
mid-below-thresh     0.319891  0.619384  0.624740  0.661975  0.668160  0.682111  0.689833  0.689777  0.673945  0.701154  0.687638
mid-normal           0.321269  0.669880  0.713333  0.707553  0.742765  0.721908  0.697291  0.746640  0.698949  0.758903  0.716206
bin-quintile         0.312592  0.574468  0.635107  0.659408  0.693003  0.698917  0.684922  0.717690  0.710775  0.760860  0.761121
bin-normal           0.322245  0.479326  0.635119  0.671364  0.659048  0.692026  0.701767  0.726629  0.761599  0.724677  0.715380
random               0.313345  0.499146  0.625866  0.662209  0.655599  0.678943  0.693644  0.676825  0.707225  0.723363  0.703220
true-random          0.314118  0.511033  0.627201  0.644138  0.691885  0.671265  0.680546  0.724677  0.729037  0.691245  0.735431
```

15%, avg (in progress on lambda2):

```
                           0         1         2         3         4         5         6         7         8         9         10
init                 0.331441  0.331441  0.331441  0.331441  0.331441  0.331441  0.331441  0.331441  0.331441  0.331441  0.331441
median-below-thresh  0.330744  0.422266  0.501311  0.525312  0.577019  0.556348  0.659990  0.693586  0.709935  0.726240  0.725822
median-thresh        0.339362  0.441407  0.484165  0.472528  0.512185  0.615415  0.681911  0.683063  0.690873  0.707963  0.679264
normal               0.325666  0.365345  0.495346  0.579724  0.607303  0.652843  0.583767  0.665412  0.704815  0.715831  0.732590
iqr                  0.326513  0.355547  0.365116  0.424651  0.612438  0.713691  0.741589  0.763744  0.766388  0.764296  0.773409
mid-thresh           0.335368  0.425620  0.614798  0.656079  0.659609  0.691863  0.746447  0.733711  0.747515  0.751883  0.746294
mid-below-thresh     0.313023  0.470647  0.484158  0.542546  0.614666  0.599603  0.629021  0.632222  0.697616  0.677381  0.734134
mid-normal           0.335689  0.526440  0.578652  0.681084  0.694374  0.676733  0.664707  0.660738  0.704870  0.735768  0.780326
bin-quintile         0.330282  0.392488  0.392286  0.622156  0.709771  0.784841  0.782455  0.773164  0.777119  0.786627  0.776614
bin-normal           0.331014  0.371036  0.362232  0.375255  0.428707  0.643257  0.701628  0.740450  0.768454  0.729100  0.772542
random               0.332925  0.367743  0.414016  0.432995  0.543384  0.634036  0.718689  0.745737  0.757653  0.772149  0.753383
true-random          0.334889  0.370553  0.353503  0.614575  0.644407  0.669578  0.756618  0.724921  0.747373  0.762847  0.760464
```

**Tuesday, September 1**

* Had AI/ML scrum meeting
* Compiled results above from overnight training
* Wrote function to examine the individual class distributions for a particular benchmark
* Fixed some data manipulation bugs
* Looked at correlations between two different test sets
  * Interesting how training duration plays a large role in precision improvement in next-batch testing, but not in the specific test set created by two sampling batches
  * Created a new measurement, average `conf_std`, to measure the spread of confidences for a particular sampling method
    * There's a weak to moderate positive correlation between this and precision improvement
  * Sticking with the rolling average benchmarks due to higher recall (~90%) compared to linear average (~75%)