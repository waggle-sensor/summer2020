# Iterative Retraining

This folder collects several disjoint scripts from other folders into one module. The goal of this pipeline is to simulate the original training of a machine learning model (based on Darknet/YOLOv3), followed by several batch iterations of training and resampling on a sample set. 

## Process

User-defined parts of this process (formalized in the parameters section) are bolded

1. An **initial data set** labeled with bounding boxes in the Darknet format is provided. This is separated into `data/images` and `data/labels` folders.
2. This data set is randomly split into training, validation, and test sets, stratified by class with **a certain proportion**
3. Data augmentation is performed on the training set, with a target of a **particular number of samples per class** to prevent class imbalance. Classes with fewer images will thus have more augmentations for a given training image. 1-2 "minor transformations" (e.g. RGB, HSV, and brightness/contrast shifts) are applied, with the option of **"major" transformations** (e.g. distortions, random crops, affine transforms).
4. Train until a certain **number of epochs** or **early stopping criteria** is met. A default criteria is implemented using known techniques (UP from Prechelt 1997), though users can implement their own.
5. Randomly split a labeled **sampling data set** into **batches of a specified size**.
6. For the first batch, run inference using the initial trained model, segmenting non-overlapping objects into individual images via bounding boxes (if performing object detection). Note: this is assumed to be done with the KAIST image data.
7. Run inference with a **fixed number of training checkpoints**, assumed to be equally spaced from prior training. This will generate a series of bounding boxes for each image; we will take the average confidence of each class label across the checkpoints and use the class with the greatest average confidence. This confidence score will be an inverse measure of uncertainty  (see Geifman et. al, 2019).
8. Using the distribution of confidences, select images to send back for retraining, with a **fixed bandwidth limit**. Although this limit will likely be file size in practice, the simulation will use a fixed number of images. See the [below section](#sampling-algorithm) for details on how this algorithm is customized.
9. Annotate/verify that the images that are "sent back" have the correct ground truth labels
10. Split this set of images into training and test/validation sets by a **certain stratified proportion**. Using more images for training is likely favorable here.
11. Incorporate the new data alongside the old data in a **fixed proportion** (e.g. 25% old, 75% new) for both the training and testing/validation sets, undersampling prior data if necessary. In the interests of saving computation time, augmentations on the old training data are retained, and new data is augmented to the extent that images reach the target number of images per class.
12. Retrain using the mixture of old and new data, benchmarking performance separately on the old test set and new test set. Use the combined test set as the stopping criteria.
13. Repeat steps 6-12 for the remainder of the batches from the large sample set.

## Configuration (Hyper)Parameters

* `class-list`: list of classes, terminated with a blank line
* `model_config`: YOLOv3 model definition file with hyperparameters
* `initial_set`: labeled directory of images for initial model training
* `sample_set`: labeled directory of images for sampling
* `train_init`: proportion of images in initial set used for training
* `valid_init`: as above, but for validation. The test set will consist of all remaining images
* `images_per_class`: target number of images per class for executing augmentation and retraining
* `aug_compose`: boolean value (0 or 1) for using major transformations alongside minor ones
* `early_epochs`: early stop value for number of (re)training epochs
* `sampling_batch`: batch size for sampling. In the SAGE implementation, this is analogous to a video stream providing a specific number of frames each hour.
* `conf_check`: number of checkpoints to use when determining confidence score
* `bandwidth`: maximum images to sample per sampling batch, across all classes
* `train_samp`: propotion of images in sample batch set to use for training. The rest will be used for testing.
* `retrain_new`: proportion of images in the revised training/test sets that will be from the sampling batch. The rest will be randomly selected from old training/test data

## Sampling algorithm

To be completed

## Output files