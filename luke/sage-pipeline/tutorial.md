# Sage Image Filtering Pipeline

## How It Works

This pipeline works in three steps:

1. Compute the brightness value of each image in the set, and find the average brightness of day images and night images, respectively. If all the images were taken at one time of day (for example, all in the daytime), the pipeline will detect this and mark the set as having only one average brightness.
2. Train either one or two background subtractors with the given images. (Only one background subtractor is trained if there is only one lighting condition.) The background subtracting model used in the pipeline does not require thousands of images to train on; it can work well even with around 50 training images.
3. Tag each image with a "foreground count" value. This value is useful to assess the likelihood that a given image is a "background image", an image with no interesting foreground features in it. This value is computed by passing an image to the background subtractor, having the subtractor extract the foreground features present in the image (giving us a foreground mask), and counting the number of foreground pixels in the foreground mask. This value by itself means nothing, but it is a adequate metric for judging the "interesting-ness" of an image. A high foreground count suggests a busy image, while a low foreground count suggests an un-interesting background image.
4. Present a few foreground count statistics to the user, prompting them to enter a threshold that will mark the images with the lowest foreground counts as background images. These tags will be written to the root folder of the input image folder in a file named `metadata.json`.



## General Usage



## Choosing a Good Threshold Value



