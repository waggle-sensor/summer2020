import os
# Explicitly setting these values at the start of the program might be necessary if there are threading/multiprocessing
# problems with your machine:
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '2'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '2'

from TrainingSet import TrainingSet
from Classifiers import TextureTemporalClassifier, TextureClassifierLBP
import pickle
import numpy as np
# import cupy as np
from matplotlib import pyplot as plt
import mydia
from sklearn.svm import SVC

N_PROCESSES = 10
plt.ioff()

def test_maximize_ensemble_accuracy():
    # Load TT classifier training with a 50fps sampling rate
    tt_classifier = pickle.load(open('test_framerate_classifiers.bin', 'rb'))[1]
    # lbp_classifier = TextureClassifierLBP()

    # Load training images
    training_obj = TrainingSet.init_many('/home/cc/ww/videos', '/home/cc/ww/masks')
    vid_paths = [vid.input_sequence for vid in training_obj.vids]
    all_x = mydia.Videos(target_size=(800, 600), to_gray=True, num_frames=3).read(vid_paths, workers=N_PROCESSES)[..., 0]
    all_y = training_obj.get_masks()
    # frames = np.frombuffer(open('frames.bin', 'rb').read(), dtype=np.uint8).reshape((-1, 60, 600, 800, 3))
    # masks = np.frombuffer(open('masks.bin', 'rb').read(), dtype=bool).reshape((-1, 600, 800))

    # Train LBP classifier
    # lbp_classifier.train(all_x, all_y)
    classify = TextureClassifierLBP()
    classify.classifier = pickle.load(open('lbp_nn.bin', 'rb'))

    # Save classifier
    # pickle.dump(lbp_classifier, open('lbp_classifier_fully_trained_%d.bin' % int(time.time()), 'wb+'))
    print('done')

def train_multiframerate_tt_classifiers():
    VIDEOS_FPS = 50
    FRAMES_TO_READ = 60
    sampling_tests = [50, 10, 2, 1]  # FPS: 50fps, 25fps, 5fps, 1fps

    # classifiers = pickle.load(open('test_framerate_classifiers.bin', 'rb'))
    classifiers = {rate: TextureTemporalClassifier() for rate in sampling_tests}
    training_obj = TrainingSet.init_many('/home/cc/ww/videos', '/home/cc/ww/masks')

    TRAIN_TEST_SPLIT = 0.9
    all_y = training_obj.get_masks()
    for rate, classify in classifiers.items():
        classify: TextureTemporalClassifier

        vid_paths = [vid.input_sequence for vid in training_obj.vids]
        reader = mydia.Videos(num_frames=FRAMES_TO_READ, target_size=(800, 600), mode=ModeFunctionFactory(rate))

        # set_start_method('forkserver', force=True)
        all_x = np.array(reader.read(vid_paths, workers=N_PROCESSES))

        # Split train and test
        split_indx = int(len(vid_paths) * TRAIN_TEST_SPLIT)
        train = (all_x[:split_indx], all_y[:split_indx])
        test = (all_x[split_indx:], all_y[split_indx:])

        # Train classifier with sampling-rate-specific training frames
        print('Training %d-fps TT classifier' % rate)
        classify.train(train[0], train[1])

        # Save classifier
        pickle.dump(classify, open('tt_classifier_%d-rate.model' % rate, 'wb+'))

        # Test classifier
        print('Testing %d-fps TT classifier' % rate)
        classify.test(test[0], test[1])

    pickle.dump(classifiers, open('many_framerates_tt_classifiers.model', 'wb+'))
    print('done')

def test_classifiers():
    # testing_set = '/home/cc/alternate_data/'
    # testset = TrainingSet([TrainingSequence(os.path.join(testing_set, vid_path),
    #                                         category=TrainingSequence.CAT_UNLABELED, fps=30)
    #                        for vid_path in os.listdir(testing_set)])

    print('Loading video frames into memory')
    testset = TrainingSet.init_many('/home/cc/ww/videos', '/home/cc/ww/masks')
    training_x = training_set.load_all(in_color=True, combine=True, dims=(600, 800), max_n=120)

    random_vids = np.random.choice(len(training_x), size=50)
    print('Training all')
    testset.train_all(training_x[random_vids], training_set.get_masks()[random_vids])

    # Restore training
    testset.classifiers = pickle.load(open('classifiers.bin', 'rb'))
    testset.classifiers.append(pickle.load(open('texture_temporal_classifier.bin', 'rb')))

    test_x = testset.load_all(in_color=True, dims=(600, 800), max_n=120, combine=True)
    testset.visualize_classifiers(train_sample=test_x)

    print('done')

def compute_on_chameleon():
    USE_PREVIOUS = True
    FOLDER_WATER_VIDEO_DATASET = '/home/cc/ww/'
    VIDEOS_FOLDER = FOLDER_WATER_VIDEO_DATASET + 'videos'
    MASKS_FOLDER = FOLDER_WATER_VIDEO_DATASET + 'masks'

    WaterSet = TrainingSet.init_many(VIDEOS_FOLDER, MASKS_FOLDER)

    # frames, masks = WaterSet.get_training_vids(n_frames=60)
    frames = np.frombuffer(open('frames.bin', 'rb').read(), dtype=np.uint8).reshape((-1, 60, 600, 800, 3))
    masks = np.frombuffer(open('masks.bin', 'rb').read(), dtype=bool).reshape((-1, 600, 800))

    ninth = int(0.9 * len(frames))
    training = (frames[:ninth], masks[:ninth])
    testing = (frames[ninth:], masks[ninth:])

    # Test out texture-temporal classifier
    ttc = TextureTemporalClassifier()
    ttc.train(training[0], training[1])
    print(ttc.test(testing[0], testing[1]))

    WaterSet.visualize(ttc, n_imgs=10)

    # Train everything
    if os.path.exists('classifiers.bin') and USE_PREVIOUS:
        classifiers = pickle.load(open('classifiers.bin', 'rb'))
        WaterSet.classifiers = classifiers
        WaterSet.texture_classifier, WaterSet.color_ggm, WaterSet.temporal_classifier = classifiers
    else:
        print('Training texture classifier')
        WaterSet.texture_classifier.train(training[0], training[1])
        print('Training color GMM model')
        WaterSet.color_ggm.train(training[0], training[1])
        print('Training FFT model')
        WaterSet.temporal_classifier.train(training[0], training[1])

        # Save training progress
        pickle.dump(WaterSet.classifiers, open('classifiers.bin', 'wb+'))

    # Train combinatorial classifier
    combine_classify = SVC(probability=True)
    # Segmentations are in shape (# vid, y-size, x-size, # classifiers)
    all_segmentations = WaterSet.segment_all(training[0])
    WaterSet.combine_classifier.train(all_segmentations, training[1])

    # Test combinatorial classifier
    test_segmentations = WaterSet.segment_all(testing[0])
    WaterSet.combine_classifier.test(test_segmentations, testing[1])

    # print('----- Visualizing classifiers -----')
    # WaterSet.visualize_classifiers(testing[0], prob_mode=True)
    # print('----- Classifier errors -----')
    # for classifier in WaterSet.classifiers:
    #     print('%s: %f' % (classifier.name, classifier.test(testing[0], testing[1])))

    # WaterSet.visualize_classifiers(n_vids=8, prob_mode=True)
    print('completed program')

if __name__ == '__main__':
    # compute_on_chameleon()
    # test_classifiers()
    train_multiframerate_tt_classifiers()
    # test_maximize_ensemble_accuracy()
