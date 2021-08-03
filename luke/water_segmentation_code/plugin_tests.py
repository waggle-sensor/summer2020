from WaterSegmentation import *

# These three models I have pickled (serialized) so that I can store their training progress. I'm not sure if it is a
# good long term solution, but it works.
model1 = pickle.load(open('tt_classifier_1fps.model', 'rb'))
model2 = pickle.load(open('tt_classifier_5fps.model', 'rb'))
model3 = pickle.load(open('tt_classifier_50fps.model', 'rb'))

# This line initializes a TrainingSet from these two folders, a water video folder and a corresponding mask folder
# Note that this only stores the paths to videos in this object; it has not yet loaded them into RAM.
data = TrainingSet.init_from_folder('/home/luke/argonne2021_data/VideoWaterDatabase/videos',
                                    '/home/luke/argonne2021_data/VideoWaterDatabase/masks',
                                    vwd_mode=True)  # Use the VideoWaterDatabase mode format

sample_vids = data.vids[96:113]  # Select some video paths from the TrainingSet object (I chose a random slice of videos)

# This loads the videos through a Python library called mydia which can load the videos straight into a numpy array. It
# is important to add this "ModeFunctionFactory" option because that tells mydia how to select frames. In this case,
# we want mydia to pull frames at 5fps to test out how the 5fps classifier fares against the others when given an input
# stream at 5fps.
vid_paths = [vid.input_sequence for vid in sample_vids]
all_x = mydia.Videos(target_size=(800, 600),
                     to_gray=True, num_frames=60,
                     mode=TrainingSet.ModeFunctionFactory(5)).read(vid_paths, workers=N_PROCESSES)[..., 0]

# Make a TrainingSet from our absolute paths. This enables us to easily test the 3 models on one test set.
# The method "visualize_classifiers" write a file "temp.png" by default to the current directory. That image shows
# the per-pixel-block prediction of the models on the sample videos.
t = TrainingSet(sample_vids)
t.classifiers = [model1, model2, model3]
t.visualize_classifiers(all_x)
