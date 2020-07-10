# Literature around water detection

- <u>A novel approach to urban flood monitoring using computer vision</u>
- <u>Fourier Transform based Features for Clean and Polluted Water Image Classification</u>
  - *Uses a 2D FFT output to classify full-image water/pollution pictures*
  - Hence, in this work, we propose a new method based on the
    **combination of Fourier spectrum and HSV color space** with
    statistical features and an SVM classifier for clean and polluted
    water image classification.
  - There are many different types of water detection strategies because there are many different types of water: ocean, river, pond, fountain, dirty, and puddle
  - Look at the **Fourier spectrum of HSV**, not RGB or grayscale
  - "It is noted from the spectrum of clean and polluted water images that brightness in HSV of clean water images is distributed in particular directions, while for polluted water images, the brightness in HSV is scattered in all the directions. In addition, the brightness in HSV of polluted water images is lower than that in HSV of clean water images. This is true because clean water surface usually contains texture with different degrees of roughness according to water types, while polluted water image contains water surfaces with irregular objects and water surfaces without texture or roughness. Therefore, one can expect **regular patterns of brightness distribution in spectrum for clean water images**, and irregular patterns of brightness for polluted water images. This observation leads to the extraction of the following features for classifying clean and polluted water images in this work."
  - Only the real part of the DFT was select for feature extraction
- <u>Floodwater detection on roadways from crowdsourced images</u>
- <u>Water detection through spatio-temporal invariant descriptors</u>
  - *Uses both texture analysis by means of Local Binary Patterns and temporal analysis by means of FFT to discern water motion vs. flag motion*
- <u>Identifying mosquito breeding sites via drone images</u>
  - Nothing to work from.
- <u>People and Vehicles in Danger - A Fire and Flood Detection System in Social Media</u>
- <u>Single Image Water Hazard Detection using FCN with Reflection Attention Units</u>
  - Major innovation is the RAU, built into the R-CNN
- <u>Research of water hazard detection based on color and texture features</u>
  - Looking specifically at off-road environment, puddles in the mud
  - Color
    - "Water body hazard in outdoor environment tend to appears higher brightness, lower saturation and lower texture than terrain around comparatively"
    - The proposed metric to separate reflective puddles from terrain is the **ratio of saturation to value**. "Therefore, we consider that ratio to be the colour-based cue to detect water region contains sky reflection"
  - Texture
    -  Angular second moment (energy) - LOWER IN WATER
      - High energy means the gray level distribution form is constant or periodic
    - Entropy (low means less texture) - LOWER IN WATER
      - The problem with this metric in my experience has been that depending on the amount of water on a field or road, puddles will become more/less reflective and this will change the entropy depending on how much sky reflection comes through.
    - Contrast - LOWER IN WATER
    - Correlation - HIGHER IN WATER
      - "Measures linear dependencies of gray level. A higher correlation shows more homogeneous in the specified orientation."
      - The paper mentioned they repeated this at 4 different orientations, upwards, downwards, left, and right
  - SVM Model trained with the above color and texture features taken from subblocks, and the best size appears to be 8-10 pixels
- <u>Daytime Water Detection Based on Color Variation</u>
  - At long distances, water reflection is the color of the sky, but as you move closer and as the angle of incidence changes, the color of the ground begins to show through more
  - "spatiotemporal analysis is useful in detecting moving water from a stationary platform, but not still water"
  - The change in **saturation-to-brightness ratio across a water body** from the leading to trailing edge is uniform and distinct from other terrain types
  - Water bodies tend to have a uniform brightness where they are not reflecting objects. **Thus, low texture can be a cue for water.** This applies only to bodies of water with substantial depth to reflect well, not puddles on the street.
  - A 5x5 intensity variance filter was used to build the metric of "intensity variance". Well-reflecting water has a low intensity variance compared to other textures. Whether this metric will distinguish low-depth water from its surroundings is unknown, although I doubt it.
  - They were able to initially grab the areas of the images that had the **lowest intensity variance** and then expand that segmentation to the rest of the water texture by also including neighboring pixels that had a **low intensity gradient**. This enabled the algorithm to initially develop a sense of where water definitely was located, and then spread out from there.
  - The bulk of the paper discusses classifying water based on incidence angle, so that could be a helpful resource in the future, but at the moment I do not want to take into account the incidence angle of a viewing angle.
- <u>Wet Area and Puddle Detection for Advanced Driver Assistance Systems</u>
  <u>(ADAS) Using a Stereo Camera</u>
  - RANSAC
    - This modeling algorithm finds a pattern in a set of observed data which specifically contain outliers. It is iterative and adjusts its model until most of the dataset is described by the model.
  - Hypothesis Generation
    - Reduces false positives
    - A mask is built by using a MoG model to model dry areas. Any regions above a certain threshold on the PDF are selected as dry regions.
  - Hypothesis Verification / Feature Extraction
    - Three features used for verification
      - Polarization difference - absdiff between horizontal polarized image and vertical polarized image (not useful for my purposes)
      - Graininess - The rate of contrast change between the original frame and blurred image with a Wiener-filter. This is helpful because dry road areas normally have a high contrast surface compared to wet areas.
      - Gradient Magnitude - The mean of the gradient image in a hypothetical window. 
- <u>Vibration-based Terrain Classification Using SVM's</u>
  - Different vibrations caused by the motion of the cart on different terrains can be classified well using just FFT output
  - Extracting features from the signal worked better for SVM classification than just piping in a 128-point FFT output
  - Features used for vibration classification which worked well in their experiment
    - Number of sign changes in the signal (a rough approximation of of the main frequency of the signal)
    - Number of traverses over the mean
    - Standard deviation of the signal
    - Autocorrelation of the signal
    - Maximum of the signal
    - The normalization of the signal (the square root of the summation of the square of each point)
    - The minimum
    - The mean
- <u>On the Segmentation and Classification of Water in Videos</u>
  - 



# Literature around plant identification

- <u>LeafSnap: A Computer Vision System for Automatic Leaf Detection</u>
  - "Leaf shape can be effectively represented using multiscale curvature measures"
    - Histograms of Curvature over Scale to extract curvature features for more accurate detection
  - Input images contain just one leaf



# Summarizing Notes

- Most common approaches to water detection
  - Initial color and texture filtering
    - MoG model to filter out dry areas on asphalt
    - Graininess metric
    - Saturation to brightness ratio for smooth, reflective water (also used is saturation to value)
    - 5x5 Intensity variance filter for water with depth that is smooth -> then segmenting neighboring regions with a low intensity gradient
    - Gradient magnitude
  - Polarization difference (requires a stereo camera)
  - Fourier spectrum for HSV (might be useful for low-depth puddle detection, but was initially intended to be used for classifying pure/polluted water)
- Open problems around water detection
  - MEDIUM: What is the minimum framerate required to discern the temporal signal of water from other motion (ex. leaves rustling, flags waving, animals/people moving, fire)
    - Answered in part by the spatiotemporal water detection paper, they had a graph that showed accuracy over time
  - MEDIUM: Apply FFT to identify more kinds of motion in a video, ex. leaves rustling or animals moving or dynamic textures
    - Cons: a lot of work has been done with dynamic textures
  - HARD: Build an algorithm that can counteract the motion of a camera in a scene so that motion (puddle disruptions, rustling leaves) can be identified using FFT
    - Cons: too much theory and geometry involved for 5 weeks