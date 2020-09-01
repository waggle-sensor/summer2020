# Iterative Bandwidth Aware Learning

This PyTorch-based pipeline simulates the original training of a machine learning model (with Darknet/YOLOv3 as the built-in architecture), followed by several batch iterations of continued learning on a sample set of images. The goal of this pipeline is to experimentally determine the most effective methods to sample images on the edge for retraining machine learning models, given limited bandwidth. It is currently configured to simulate this process, without actually deploying code on edge devices, though its components can be easily modified for production.

To run the main module for sampling and retraining (and optionally training an initial model), execute the following:

```
python3 . --config <config file> [--reload <checkpoint model>]
```

A [configuration file](#configuration-parameters) with various parameters (see below) must be created and specified with `--config`. 

If the `--reload` option is passed in with checkpoint model weights (matching the given model), the initial training process will be skipped, and sampling/training will begin immediately. Note that checkpoint model should follow the appropriate [naming scheme](#training-output) to prevent errors.

## Process Overview

User-defined parts of this pipeline (formalized in the parameters section) are italicized

1. An *initial data set* labeled with bounding boxes in the Darknet format is provided, based on a *provided class list*. This is separated into `data/images` and `data/labels` folders.
2. This data set is randomly split into training, validation, and test sets, stratified by class with *a certain proportion*
3. Data augmentation is performed on the training set, with a target of a *particular number of samples per class* to prevent class imbalance. Classes with fewer images will thus have more augmentations for a given training image. 1-2 "minor transformations" (e.g. RGB, HSV, and brightness/contrast shifts) are applied, with the option of *"major" transformations* (e.g. distortions, random crops, affine transforms).
4. Train until a certain *number of epochs* or *early stopping criteria* is met. A default criteria is implemented using known techniques (UP from Prechelt 1997), though users can implement their own.
5. Randomly split a labeled *sampling data set* into *batches of a specified size*.
6. For the first batch, run inference using the initial trained model, segmenting non-overlapping objects into individual images via bounding boxes (if performing object detection). Note: this is assumed to be done with the KAIST image data.
7. Run inference with a *fixed number of training checkpoints*, assumed to be equally spaced from prior training. This will generate a series of bounding boxes for each image; we will take the average confidence of each class label across the checkpoints and use the class with the greatest average confidence. This confidence score will be an inverse measure of uncertainty  (see Geifman et. al, 2019).
8. Using the distribution of confidences, select images to send back for retraining, with a *fixed bandwidth limit*. Although this limit will likely be file size in practice, the simulation will use a fixed number of images. See the [below section](#sampling-algorithm) for details on how this algorithm is customized.
9. Annotate/verify that the images that are "sent back" have the correct ground truth labels
10. Split this set of images into training and test/validation sets by a *certain stratified proportion*. Using more images for training is likely favorable here.
11. Incorporate the new data alongside the old data in a *fixed proportion* (e.g. 25% old, 75% new) for both the training and testing/validation sets, undersampling prior data if necessary. In the interests of saving computation time, augmentations on the old training data are retained, and new data is augmented to the extent that images reach the target number of images per class.
12. Retrain using the mixture of old and new data, benchmarking performance separately on the old test set and new test set. Use the combined test set as the stopping criteria.
13. Repeat steps 6-12 for the remainder of the batches from the large sample set.

## Configuration Parameters

These are stored in [`retrain.cfg`](./config/retrain.cfg) (or moved to the file of your choice) and should remain consistent for a given training and sample set

**Overall Parameters**

High-level parameters affecting both initial training and retraining

* `class_list`: list of classes, terminated with a blank line
* `model_config`: YOLOv3 model definition file with hyperparameters
* `images_per_class`: target number of images per class for executing augmentation and retraining
* `aug_compose`: boolean value (0 or 1) for using major transformations alongside minor ones
* `early_stop`: boolean value to determine if early stopping will be used
* `max_epochs`: maximum number of (re)training epochs, if the early stop criteria is not reached
* `conf_check_num`: (maximum) number of checkpoints to use when determining confidence score
* `logs_per_epoch`: number of times loss and its associated parameters are logged per epoch, between batches. This is linearly distributed throughout an epoch.

**Initial Training**

* `initial_set`: labeled directory of images for initial model training
* `train_init`: proportion of images in initial set used for training
* `valid_init`: as above, but for validation. The test set will consist of all remaining images

**Sampling and Retraining**

* `sample_set`: labeled directory of images for sampling
* `sampling_batch`: batch size for sampling. In the SAGE implementation, this is analogous to a video stream providing a specific number of frames for a certain time interval (e.g. 1000 frames per hour)
* `bandwidth`: maximum images to sample per sampling batch, across all classes
* `train_sample`: propotion of images in sample batch set to use for training.
* `valid_sample`: as above, but for validation. The rest will be used for testing.
* `retrain_new`: proportion of images in the revised training/test sets that will be from the sampling batch. The rest will be randomly selected from old training/test data
* `parallel`: boolean value to determine if GPU parallelization and multithreading will be used when running the sampling/retraining and benchmarking pipelines, with multiple sampling methods in parallel. 

**Output Folders**

* `log`: directory for logs created during training
* `output`: directory for sample splits, benchmarks, and other files for analysis
* `checkpoints`: directory for output models

**UP stopping criteria**

* `strip_len`: range of epochs to check the validation loss on
* `successions`: number of successive strips with strict increase of loss before training is stopped

**Hyperparameters**

Parameters for basic YOLOv3 models, used for initial training, retraining, and benchmarking:

* `img_size`
* `batch_size`
* `clip`: normalized value for gradient clipping
* `gradient_accumulations`
* `evaluation_interval`
* `checkpoint_interval`
* `multiscale`
* `n_cpu`
* `iou_thres`
* `nms_thres`
* `conf_thres`: minimum class confidence for a single detection to be counted when performing non-max suppression
* `pos_thres`: minimum confidence (object and class predictions multiplied) for a detection to be considered a positive result. Modify this as needed based on confidence score distributions in benchmarks.

## Ground Truth Function

Aside from the configuration file, the [`label_sample_set(img_path)` function in `userdefs.py`](./userdefs.py#L16) must be correctly completed for the pipeline to work as expected. Because the pipeline is meant to simulate benchmarking and sampling at the edge alongside labeling and retraining on a cloud server, it is not assumed that sample images are already labeled with the ground truth. The `label_sample_set` function thus returns a list of labels in [Darknet format](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects), given an image path.

If your dataset is already cleaned and labeled, with labels having a `.txt` extension instead of an image extension and located in a `labels` folder, there is no need to modify the function.

## Training Output

* Checkpoints: model weight state dictionaries from training models across all sampling methods, with an output frequency specified by `checkpoint_interval`. 
  * Located in the specified checkpoint directory, with a file format of `<prefix>_ckpt_<epoch number>.pth`. The prefix is `init` if training a model without a starting baseline, and prefixes take on the name of their sampling method (provided in `userdefs.py`) when retraining. 
  * Epoch numbers are *continuous* from the baseline epoch number, such that if we reload from `init_ckpt_50.pth` with the median threshold sampling method, `median-thresh_ckpt_51.pth` will be the next generated checkpoint.
  * Checkpoints must be properly named for the analysis tool and benchmarking for sampling to work as expected
* Batch splits: batches of images from the sample set used for benchmarking and retraining, with a size based on `sampling_batch`
  * Splits are plain text lists of image paths, with a file format of `sample<batch number>.txt`. The split number starts from 0
  * If the sample set contains leftover images (i.e. `sampling_batch` doesn't evenly divide the number of labeled images), the last batch split with fewer images will be generated but will not be used for sampling
  * Batches are randomly generated, without stratifying by class
* Benchmarks: CSV files containing the inferencing results from averaging a set of linearly-spaced models generated prior to the batch split we are inferencing on. This file is named `<sampling method><batch number>_benchmark_avg_1_<last epoch>.csv`.
  * Columns contain the image path for an inference, the detected label, the ground truth label (if available), output confidence as an average of all detections, standard deviation of the confidences when taking the average, and if the detection was a "hit" (true positive or false negative, in contrast to FP/TN)
  * If the ground truth bounding box doesn't overlap with the detected box by more than `iou_thres`, the ground truth and inferred label will be listed in two separate rows
  * Different types of benchmarks may be generated with the analysis tool, aside from the one output by training
* Sample sets: unlabeled lists of images that are "sent back" from the edge based on benchmark results, named as `<sampling method><batch number>_sample_<epoch number>.txt`. These are later split into train/validation/test sets
* Train/validation/test splits: plain text files of image paths containing splits, generated after ground truth is known, with a file format of `<sampling method><batch number>_{test,train,valid}.txt`.
  * Note that images from the same batch may appear in the validation set of one sampling method but the training set of another sampling method due to the pseudo-random nature of sampling and the iterative stratification algorithm
  * Within the same sampling method, images will never appear in two sets, even across batches



## Analyzing Results

The exported benchmark `.csv` files on each batch set can be interpreted with any standard spreadsheet software, while the generated checkpoints can be loaded with PyTorch with the given Darknet model parameters and used for reference.

For instructions on how to use the provided `analysis.py` module, please [refer to this document](./README-analysis.md). 

## Augmentation Pipeline

The augmentation and sampling piplines may be modified via functions in `userdefs.py`, similar to the ground truth function. The current pipeline relies on the [Albumentations library](https://github.com/albumentations-team/albumentations) for augmenting images for (re)training, as it provides a flexible way to transform bounding boxes alongside images. 

A "minor" augmentation is applied on each image, with a probability of a "major" one. These can be modified in `get_augmentations()`. Note that the current transforms rely on a [fork of the library](https://github.com/spencerng/albumentations) to provide some bounding box transformation functions.

If you want to modify the entire augmentation pipeline with your own functions, feel free to modify [`augment()` and `augment_img()` in `augment.py`](./retrain/augment.py#L51).

## Sampling Algorithm

Various sampling approaches are defined in [`sampling.py`](./retrain/sampling.py) and can be used to create your own sampling functions. These functions take an input `ClassResult` (after running inference on a batch of images) and return a list of images that are sent back from the edge for training and testing (called the sample set). The included methods use the confidence of an image's inferred label(s) as the basis of its selection into the sample set.

In each method, we define a probability density function (PDF), P(*x*), that determines the likelihood of selecting a label with confidence *x* into the sample set. For example, we may have a uniform distribution where P(*x*) = 2 for 0 < *x* <= 0.5 and P(*x*) = 0 elsewhere, or we could have a normal distribution function. This function is used in either **probability sampling** or **bin sampling**. Note that if an image contains multiple classes, all of its labels will be used for retraining if it is selected to be in the sample set.

### Probability sampling

We randomly select images to include in accordance with the PDF, until the sample set size limit (bandwidth) is reached or there are no more images in the batch with a confidence score that can be selected. This process is stratified by inferred class by default, though there are options to ignore class balancing. Sampling methods that incorporate this approach include:

* In-range sampling: this generates a uniform distribution for [*a*, *b*), where 0 <= *a* < *b* <= 1
* `random`: a baseline for retraining, where we use in-range stratified sampling on the interval [0, 1). 
* `median-thresh`: we compute the median confidence per class after running inference on the batch set, then in-range sample along [median, 1) and [0, median) (in the `median-below-thresh` variant)
* `iqr`: we compute the first and third quartile confidences per class, then sample along [Q1, Q3)
* `mid-thresh`: we sample along either [0, 0.5) (in the `mid-below-thresh` variant) or [0.5, 1), stratifying by class
* `normal`: instead of a uniform distribution, we use a normal density function centered at the mean confidence of each class and with a standard deviation following that of the class's labels
* `mid-normal`: we use a normal PDF centered at 0.5 with a standard deviation of 0.25

### Bin Sampling

In this method, equal-sized confidence bins (e.g. [0.0, 0.2), [0.2, 0.4), etc.) are created in order to sample from a range of different confidences instead of randomly sampling individual images. The number of samples collected for each bin is proportional to the area under the PDF curve within that bin, with undersampling as needed. Sampling is also class-agnostic, so we aim to improve the classes that generally the least precision.

Sampling methods that incoporate this approach include:

* `bin-quintile`: intervals of 0.2 are created, with a uniform PDF. This aims to collect equal numbers of each bin
* `bin-normal`: a normal PDF centered at 0.5 with a standard deviation of 0.25 is used. This attempts to eliminate the slight skews in the sample (propagated from the original confidence distribution) that can be present in the mid-normal method, as images with confidences slightly deviating from 0.5 still have a very high chance of being selected.

### Adding Sampling Methods

In [the `userdefs` module](./userdefs.py), a dictionary of sampling methods are returned from `get_sample_methods()`. Each entry contains a function-argument pairing, where the sampling function returns a list of image paths to sample, given a [`ClassResult`](./analysis/results.py#L80) object as the first argument. This data object contains an ordered dictionary for each bounding box in either a particular class or the entire sample set (as denoted by its `name` attribute), with entries for its ground truth, confidence, predicted class, and image path.

The prototype for your custom sampling function should resemble the following:

```
def custom_sample(result, params ...):
	sample_pool = result.get_all()
	chosen_samples = []

	# This can loop through samples once, iterate until a target number is reached, etc.
	for sample in sample_pool:
		conf = sample["conf"]

		# Condition for selecting an image
		if ...:
			chosen_samples.append(sample["file"])
	return chosen_samples
```

If the function returns more images than the number specified in the configuration `bandwidth`, `create_sample()` will randomly remove images to enforce the limit, stratifying by class if specified.

### Extending Sampling Methods

The arguments in the functions provided in `sampling.py` also allow for flexibility without needing to create your own sampling methods from scratch. For example, the function `in_range_sample(results, min_val, max_val)` allows you to create simple cutoffs, which you can place in the dictionary returned in `get_sample_methods()`.

The function `prob_sample(result, desired, prob_func, *func_args, **func_kwargs)` also allows you to define your own probability curve `prob_func`, which takes in an input confidence in [0, 1) and returns the probability of choosing that image. By listing `prob_sample` in the dictionary along with the appropriate arguments, you can easily sample along any probability density function.


## Architecture Modifications

Many of the pipeline's scripts rely on a modified [PyTorch implementation of YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) included in the `yolov3` package. However, if you want to expand it to other image network architectures, you may want to modify [`benchmark_avg()` in `benchmark.py`](./analysis/benchmark.py#L147) and its related functions to create a similar dataframe of results with inferred labels and confidences. Training functions in the [main module](./__main__.py) and [`retrain.py`](./retrain/retrain.py) would also have to be substituted, though sampling functions could remain unchanged.

For sampling other types of data beyond images, you will need to modify the `ImageFolder` and `LabeledSet` class objects in [`dataloader.py`](./retrain/dataloader.py) to generate splits and fetch data.

## Further Information

For more scientific background on this learning pipeline and how its various aspects were conceived, check out [this article on the Sage website](https://sagecontinuum.org/science/bandwidth-aware-learning/).

Feel free to also contact me at spencerng [at] uchicago [dot] edu with any questions or comments!