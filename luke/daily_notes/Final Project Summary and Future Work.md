# Final Project Summary and Future Work

### Intro

Hello there! Ah yes, I see you are also another water segmentation enthusiast. This is a welcoming place. This place includes most - if not all - of the lessons I have learned on the long, winding road that is computer vision-based water segmentation research. You will experience hardship when reading this of my journey. You might weep as I recount the long days of toil that have only amounted to spagetti code and high training loss. But do not fear. You, my fellow traveler, are my only hope. It is to you that I pass the baton. Go and learn my friend...

### Color Detection

Color-based segmentation performs poorly by itself (since water can take on a variety of colors in different situations), but it can be used to *extend* a hypothesis region of water. What this means is that color should not be used as the primary cue for generating region hypotheses, but that, once a hypothetical region is generated, color gradients can be used to "lasso in" the remaining edges of a body of water that may be overlooked by the temporal or texture classifier. This is the technique partly employed by the authors of "Daytime Water Detection Based on Color Variation", a paper which looks at the color properties of water to generate per-pixel predictions of water regions.

![color_hypothesis_region_fig](images/color_hypothesis_region_fig.png)

As you can see in the middle column of the last row of the above image, the discriminating feature of intensity variance was able to segment the further edges of the puddle. The closer regions which exhibited different color properties due to the reflection of light were not included in this segmentation. This is a problem that also occurs with the temporal classifier (which I will discuss in more detail in a later paragraph). The main role of the color classifier, in my opinion, should be to fill in the gaps left by the other two more-sophisticated classifiers. 

Although I have not experimented with this method, to increase accuracy and to make sure all regions of water are included in the segmentation, one could use a color gradient method to augment the existing segmentation to include portions of water that are left out of the segmentation while not including any non-water regions. This is the technique adopted by the previously-cited paper. 

Practically, this could look like computing the grayscale gradient at the edge of the segmentation mask and then having the mask expand if the color gradient is neglible. This would work like a *more intelligent floodfill*, since it would start at the edges of the segmentation mask and work outwards. There would have to be a level of thresholding applied to the gradient, and this technique would not work well if there were dark reflections on the water, since that would result in a large color gradient, even though we would want to include that area of water.

A segmentation algorithm which looks at just color features will be able to identify smooth water at far distances (like the above figure), but this technique can only get you so far. Oceans and rivers have changing brightness values depending on the surface ripples and waves, and any amount of reflections will confuse a color-based classifier. Take for example this image which shows water which reflects light in some areas and dark sky in other areas:

![rain_ex1](images/rain_ex1.jpg)

In the areas where light is beaming straight towards the camera, an intense vertical reflection is observed on the puddles. A paper named "Single Image Water Hazard Detection using FCN with Reflection Attention Units" tackled this problem. They proposed a new neural network which was able to intelligently match the reflection with the original light source:

![rau_fig](images/rau_fig.png)

This enhanced the network's ability to identify puddles, since if it could detect a reflection, water almost certainly existed at that spot. (Looking into the design of this neural net could be of help to design a net for general-purpose water segmentation, not just with puddles as in the abovementioned paper)

To summarize, color-based segmentation shows promise in the areas where texture and temporal-based segmentation fail, specifically in segmenting smooth, distant regions of water.

### Texture Classifier

A more sophisticated approach to water identification uses Local Binary Patterns (LBP's). This is done in the paper named "Water detection through spatio-temporal invariant descriptors". In this paper, the rearchers applied a 3x3 LBP kernel to many images of water and non-water. For each LBP image they extracted a patch (of unmentioned size) from water and non-water images. They found that the following shapes correlated positively and negatively with water:

![texture_features](images/texture_features.png)

They further explained why these shapes correlated positively and negatively with water.

One concern that I have is that a R-CNN could be able to achieve this level of feature extraction automatically, without the LBP processing step. I don't know how accelerated the LBP step could be on a device like the NVIDIA-NX, but it might be slower to convert a whol e image into LBP format, split the image into 10x10 blocks, and then pass those blocks into a more simplistic classifier like a Random Forest or an SVM. Using a R-CNN from the start could very well be the best option.

![rf_classify_lbp_features](/home/ljacobs/Argonne/water_pipeline/results/rf_classify_lbp_features.png)

![texture_predictions_row](images/texture_predictions_row_better.png)

The texture classifier is especially good at identifying water with large ripple/wave features. It also performed quite well at discriminating trees against a sky background from reflected trees (last two columns). It turns out that it is quite bad at discriminating fire, but that's an especially difficult problem. One thing that I have noticed is that the texture classifier can overfit on water that has ripples in it and learns to ignore smooth water. That's a problem.

One future direction to take with texture classification is to look into Gray-Level Coocurrences Matrices (GLCM) which can generate a bunch of metrics about the texture of a region. I tried a metric

### Temporal Classifier

The temporal classifier is one of the most sophisticated classifiers that I have worked with. It relies on the FFT for the bulk of its processing. There are a few steps to how this classifier works, and they are explicitly laid out in the paper "Water detection through spatio-temporal invariant descriptors". In the paper, a sequence of grayscale frames were split into many smaller 3x3 patches which were then averaged along the spatial axes, resulting in a lot of time-based signals. These signals then were the features input into an SVM which could identify each 3x3 patch as water or not for a certain amount of frames.

![example_fig_classification_decay](images/example_fig_classification_decay.png)

The above image shows the creeping "decay" of confidence in the temporal classifier's predictions as it tries to segment water that is further from the camera view. (The original image is on the left and the predictions are on the right - more yellow means more confident.) It makes sense that the temporal, or motion-based, classifier is better able to segment water that is more visible and closer to it, since it segments water based on surface ripples and waves. 

### Inferencing

`WaterSegmentationInference.py` is unfinished. I did not get the time to get around to deploying the classifiers with command line support, but if you look at `WaterSegmentation.py`, there should be a function or two that gives an example of how inference happens.

Inference is done through the `segment` method of a Classifier object, which can accept multiple frames as one inference point or even multiple inference points each containing multiple frames. This means if you want to perform a single mask segmentation, you would want to pass an ndarray in the shape of (1, number of bundled frames, y-size, x-size, [HSV if necessary]) where the first dimension is 1. 

I have not optimized the method for streaming from a device like a Sage node, so the inferencing code should be adapted to optimize for that. Also, I started to look into a library named CuPy, which is a GPU-based version of Numpy that would likely work faster on the NX than normal Numpy, so that might be something worth looking into, especially for inferencing.

As for combining the results of the multiple classifiers, it appears that accuracy can be improved by training an SVM to weigh the probability outputs of each pixel (a feature vector of length 3, one for each classifier output) automatically. This might add some extra overhead, however. The best method I found for combining the output of the texture classifier and the temporal classifier was just to make a hybrid texture-temporal classifier. This hybrid classifier applies the same preprocessing steps as the temporal classifier, but on a sequence of LBP-processed frames. From the short experiment I did with it, it seems to work better than either the texture or temporal classifier.