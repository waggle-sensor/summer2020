# What is the minimum framerate required to obtain good water segmentation results?

- If there is no substantial accuracy boost in feeding high-fps frame sequences into the texture-temporal classifier, why not just use low fps video?
- I trained 4 identical texture-temporal classifiers over 234 training videos sampled at different framerates (1, 5, 25, and 50 fps values) to see if the classifiers trained on high-fps footage performed substantially better than the classifiers trained on low fps video.
- I held the number of input frames constant to give the same feature vector length to each classifier. (This does mean that the 1fps camera is viewing water changing over 60 seconds, as opposed to the 50fps camera which is viewing water changing over course of a little over 1 second, but at a higher fps)
- For testing I gave 26 videos to each classifier at the sampling rate they were trained on (passing 60 frames as the feature length). 

```
FPS                      | 1fps  | 5fps  | 25fps | 50fps |
Per-Pixel % Median Error | 37.00 | 38.00 | 33.00 | 35.00 |
Per-Pixel % Mean Error   | 39.25 | 40.06 | 37.26 | 38.90 |
```

- Error values were computed by taking the absolute difference between ground truth mask frames and the segmentation probability predictions (each pixel representing a probability that it was water) for the 26 testing videos
- Although the error metrics are not very clear and further investigation is necessary, it does not seem that fps sampling rate has a significant effect on the prediction ability of the texture-temporal classifier.
- This implies that a texture-temporal classifier running on a node should only need to sample images once every second for 60 seconds to get a prediction of water based on motion.
- 

