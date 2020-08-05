# Retraining Machine Learning Models from the Edge

Hi! I'm Spencer Ng, a rising undergraduate second-year studying Computer Science and Theater & Performance Studies at the University of Chicago. 

As seen throughout this website, Sage is a platform that scientists can use to deploy machine learning models at the "edge" and track both human-centered and natural features around the world. These include smoke from wildfires, floods, snow, vehicles, and more. Models are generally created by learning from a specific set of data, then they are used on Sage nodes in a variety of locations. As a result, they might not be as accurate in the real world compared to test conditions. 

One way to resolve this is to give a model a greater variety of data to learn from, so it can generalize to more environments or adapt to a specific one. But what happens if there aren't enough existing data to train an accurate model? With the help of the original model, we would need to collect new data (e.g. images and videos) containing the features we are tracking from the nodes at the edge. These data would be sent to a central server (the Beehive), which would then retrain models and send them back to the edge. This continuous approach can be applied to improve the performance of a variety of machine learning techniques used on Sage.

However, a major limitation is the rate at which data can be sent from the nodes, as Sage relies on a cellular data connection, which can be slow and/or expensive to use, depending on the environment. Consequently, nodes receive and process more data at a given time period than they can send. My project aims to empirically determine the most effective strategies to sample data for retraining machine learning models with improved performance.

I first considered information available on data at the edge and ways to measure performance that would be scalable to an array of experiments run on Sage. In general, machine learning models output a *prediction* and a *confidence score* about that prediction when given input data. For example, a model that detects car make/model and has an input video feed of a street intersection would draw bounding boxes around objects that it thinks are cars and label them with a particular confidence (e.g. a Toyota Camry at 85% confidence; insert picture above of this). This inferencing occurs at the edge, so we would be able to use model outputs as parameters for creating a sample. Confidence can be viewed as a "negative measure" of uncertainty, and it made intuitive sense to sample data that models are more uncertain of, as they would be more likely to be incorrectly labeled. Retraining with previously-inaccurate data (after telling the model their true labels) could then make the model more accurate in the future.

## Making Uncertainty More Certain

To evaluate this assumption between uncertainty and accuracy, I trained a simple object detection network using YOLOv3 to recognize 12 different alphanumeric characters using the 74K Character Dataset and ran inference on the relevant images from the KAIST Street Sign dataset. The initial results supported the idea that output confidence was positively correlated with the accuracy and precision of the model's results, but a histogram of the confidences revealed that both accurate and inaccurate detections tended to have high confidences. This would make it difficult to differentiate between data that are correctly and incorrectly labeled at the edge using only their confidences, as we would not reliably know their "ground truth." I then implemented an improved measure of uncertainty suggested by (insert authors here) in hopes of differentiating the confidence distributions for accurate and inaccurate labels. In this method, each input datapoint (e.g. an image) is evaluated by a series of checkpoint models generated in the process of training the latest model, outputting many bounding boxes and confidences that are combined and averaged through non-maximum suppression. This new inferencing method yielded confidence distributions that were more left-skewed for accurate detections and right-skewed for misses, making it easier to sample images that tended to be inaccurate.

A variety of sampling methods based on the overall confidence distrubtion at the edge were then tested. Although we wanted to bias the selection towards "hard" cases with lower confidences (and therefore boost their accuracy and confidence score), it is also important to retain some "easy" cases in the model to maintain its overall accuracy(citation needed). Thus, our sampling procedures included the following:

* Taking all data with confidences above the median confidence
* Taking all data below the median confidence
* Sampling data according to a normal probability density function centered at the mean confidence
* Taking all data within the interquartile range of confidences
* Sampling data according to a normal curve centered at 0.5
* Taking all data with a confidence between 0.25 and 0.75
* Taking data uniformly in 0.2 confidence increments

In each case, the confidence distribution of each inferred class/label type (e.g. a specific car make/model) is independently generated, then data are sampled. That way, the final sample have a roughly representative of the various data classes given to the model; a car type that tends to have lower confidences compared to the overall median confidence would still be included in the sample if using the first sampling method. The "bandwidth limit" was then enforced after a sample was created by randomly removing images for each class until the sample had an equal number of images per class, with a total number of images under the limit.

## Testing Cycle

In a practical implementation of the sampling and retraining process on Sage, data would be continuously collected and sent to the Beehive periodically (e.g. twice a day) after running the inference and sampling functions at the edge. Once retraining is completed, the new and improved models would be sent back to Sage nodes and used for inference. To simulate this, 3,500+ images of the 12 different character classes from the KAIST dataset were split into batches of 500 images each, of which at most 300 were sampled and "sent back" for retraining. The retrained model would then be used to sample the next batch, until the 7 batches were all used. The goal was to track changes in the results of inferencing on a subset of both initial and retraining data reserved for testing as training continued; there could be increases in precision within each iteration, throughout the entire retraining process, or perhaps decreases due to biasing the model on "hard" data. These analyses were run for the various sampling methods described above.

[Results here]

A second method to track improvement was to use the inferencing results on the 500-image batches themselves, the inputs to the sampling functions, which were controlled between each sampling method. On the other hand, the pseudo-random method of enforcing the bandwidth limit meant the test sets generated from each iteration sample varied in the number of images they contained and their actual images. Comparing how well the model at the end of each iteration inferred the ground truth for its upcoming batch would thus yield a more consistent test set between sampling strategies.

[Results here]

## Pipeline Development and Usage

[Insert graphic]

The software pipeline for iterative retraining is meant to be scalable and easy for users to modify and deploy. An initial labeled set of data, separate from the data used for retraining, is first split into training, validation, and testing sets, which are used to create a deep neural network. The training set is then augmented via a series of random affine transforms and shifts from the Albumentations library, such that each class contains the same number of training images. These images are input into YOLOv3, a popular object detection framework, and training continues until the UP3 early stop criteria is reached, described by (authors) as when validation loss increases three times in a row for a particular strip of training epochs. It is also possible to initialize the network with a set of pretrained weights or modify the pipeline to use other types of neural networks.

Once an *initial model* is created, inference is run on the first batch of images using non-max suppression and a series of checkpoint models, as described above. These results are used to create a sample, in accordance to a sampling function that takes in images' confidences and selects a subset of the batch. The batch is then annotated with the ground truth (automatically done in the simulation), split into three sets again, and a proportion of seen data is mixed in with the sets derived from the sample. The goal of retaining some of the old data beyond existing weights was to determine if we could create a more generalized model, without biasing the model too heavily on the sample data. Retraining then occurs with the new retraining set using the same parameters and processes, except the most recent model is used as initial weights. This cycles continues in the simulation run until there are no more complete batches of images. In a real-world implementation, analysis on the model's precision and accuracy after each iteration could be done to manually stop the pipeline.

## Future Work

The sampling and retraining pipeline's results look promising, and further studies could [insert example cases that aren't done by the end for generalizing and to support found methods]. There could also be better support for generalizing the pipeline to retraining on non-visual data, such as text and time series sensor readings. [more to come, depending on how code is refactored]

3) overview of the Waggle/Sage approach for addressing the problem at the edge (high level, non-expert) (for example, camera images are processed every 5 seconds and a deep learning algorithm is used to detect cars in the image and then calculates.... and the results are sent to a central server (the beehive),,, etc.).  Include a little about how training is done.   



4) The technical approach.  

Technical details here (photos, diagrams)

5) Next Steps. 

Pictures and diagrams should be included. Then at the bottom, we can add your student photo, links to your personal web page or LinkedIn or University page, your poster if you did one, etc. 

1500-2000 words
