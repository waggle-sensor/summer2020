from typing import List
from TrainingSequence import N_PROCESSES
from skimage.feature import local_binary_pattern
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, LinearSVC
from matplotlib import pyplot as plt
from Utilities import *
import joblib


class Classifier:

    """
    This is the base class for the rest of the classifiers that I have made.
    """

    def __init__(self, name: str, img_classify=True, n_frames=1):
        self.name = name
        self.img_classify = img_classify  # To signify whether a classifier is an image classifier or a video classifier
        self.n_frames = n_frames

    def train(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError('Method not implemented')

    def segment(self, x: np.ndarray, prob_mode=False):
        """
        Segment a list of videos given a sample frame sequence (or single frame) from each of them. Output shape will
        be (# video, y-size, x-size)
        :param x: Input array
        :param prob_mode: Whether to output probability maps or binary predictions for each segment
        :return: An array of masks for each input video
        """
        if (4 > len(x.shape)) or (len(x.shape) > 5):
            raise RuntimeError('Unable to accept shape length %d, only 4 or 5 (length of 5 will be converted '
                               'if necessary)' % len(x.shape))

        raise NotImplementedError('Method not implemented')

    def test(self, testing_x: np.ndarray, testing_y: np.ndarray, prob_mode=True, output=True) \
            -> Tuple[float, float]:
        """
        Test this classifier with a testing set.

        :param testing_x: Array of frames (# vid, # frames, y-size, x-size, [HSV channels])
        :param testing_y: Array of masks
        :return: Stats about error
        """
        results = self.segment(testing_x, prob_mode=prob_mode)
        error = abs(results.astype(np.float64) - testing_y.astype(np.uint8))
        stats = float(np.median(error)), float(np.mean(error))
        if output:
            print('[TEST] %s\n\tMedian Error: %f\n\tMean Error: %f' % (self.name, stats[0], stats[1]))
        return stats


class TextureClassifierLBP(Classifier):

    TEXTURE_BLOCKSIZE = 10

    def __init__(self):
        super(TextureClassifierLBP, self).__init__('lbp texture')
        # self.classifier = RandomForestClassifier(n_jobs=-1)
        self.classifier = MLPRegressor()

    @staticmethod
    def lbp_process_frames(sample: np.ndarray) -> np.ndarray:
        """
        Processes a sample of frames and outputs the LBP of those frames.

        :param sample: Sample of LBP frames in shape of (# frames, y-size, x-size)
        """

        LBP_RADIUS = 3
        processed_frames = np.zeros(sample.shape, dtype=np.uint8)
        for i in range(len(sample)):
            processed_frames[i] = local_binary_pattern(sample[i], LBP_RADIUS * 8, LBP_RADIUS, method='uniform')
        return processed_frames

    @staticmethod
    def lbp_process_frames_multiproc(frames: np.ndarray, prefer='processes'):
        """
        A function to assign workers the task of processing the LBP descriptions of frames in the shape of:
          (# of frames, frame-y, frame-x)
        This functions processes every pixel in the frame array and returns the result of the computation as an array of
        the same shape, but each item being the result of a LBP computation.
        """

        if len(frames.shape) < 3:
            raise RuntimeError('Incorrect dimensions for LBP input')

        frames_to_process = np.array_split(frames, N_PROCESSES)

        print('Computing LBP of %d frames split among %d processes...' % (len(frames), N_PROCESSES), end='')
        results = joblib.Parallel(n_jobs=N_PROCESSES, prefer=prefer)(
            joblib.delayed(TextureClassifierLBP.lbp_process_frames)(sample) for sample in frames_to_process)
        print('done')

        results = np.vstack(results)

        return results

    def train(self, x: np.ndarray, y: np.ndarray, max_blocks=1000000, prefer='processes'):
        """
        Train texture classifier.

        :param x: Array of frames (# videos, # frames, y-size, x-size, [HSV channels])
        :param y: Array of masks (# videos, y-size, x-size)
        :param max_blocks: Max 10x10 LBP blocks to train on
        :return:
        """

        if len(x.shape) > 4:
            print('Converting 3-channel HSV images to grayscale')
            x = x[..., 2]

        n_vids, n_frames, y_img, x_img = x.shape

        # Account for training multiple frames per video

        x = x.reshape((-1, y_img, x_img))  # If many frames are given per video, flatten them

        samps_labeled_frames = np.array_split(list(zip(x, y)), N_PROCESSES)

        def vid_worker(labeled_frames: List[Tuple[np.ndarray, np.ndarray]], blocksize) -> Tuple[np.ndarray, np.ndarray]:
            """
            Accepts list of VideoCapture paths (str) and returns a tuple of two ndarray's, one the array for water
            blocks and the other for no water blocks.
            """
            print('Starting worker')
            w_lbp_blocks = []
            nw_lbp_blocks = []
            for frame, mask in labeled_frames:
                # Process frame and mask into blocks
                samp_lbp_img = TextureClassifierLBP.lbp_process_frames(frame[np.newaxis, ...])[0]
                lbp_blocks, _ = sliding_win_select(samp_lbp_img, blocksize)
                mask_blocks, _ = sliding_win_select(mask, blocksize)

                for lbp_blk, mask_blk in zip(lbp_blocks, mask_blocks):
                    if mask_blk.sum() == int(blocksize * blocksize):
                        w_lbp_blocks.append(lbp_blk)
                    else:
                        nw_lbp_blocks.append(lbp_blk)

            return np.array(w_lbp_blocks).reshape((-1, blocksize ** 2)), \
                np.array(nw_lbp_blocks).reshape((-1, blocksize ** 2))

        # Begin threads and then concatenate each worker's result data
        results = joblib.Parallel(n_jobs=N_PROCESSES, prefer=prefer)(
            joblib.delayed(vid_worker)(work, self.TEXTURE_BLOCKSIZE)
            for work in samps_labeled_frames)
        all = np.array(results)

        # Split training data into water and non-water patches
        w = np.array(np.concatenate(all[:, 0]))
        nw = np.array(np.concatenate(all[:, 1]))
        np.random.shuffle(w)
        np.random.shuffle(nw)

        # Train Random Forest with a specified number of patches
        w_pts = min(max_blocks, len(w))
        nw_pts = min(max_blocks, len(nw))
        print('Fitting RF classifier...', end='')
        self.classifier.fit(np.vstack((w[:w_pts], nw[:nw_pts])), np.concatenate((np.ones(w_pts), np.zeros(nw_pts))))
        print('DONE!')

    def segment(self, x: np.ndarray, prob_mode=False, avg_frames=2):
        """
        Returns a binary mask of the same size as the input image (with padding) where white pixels are sections of
        water identified by the internal classifier. If multiple frames are given the probability maps of each will
        be averaged.

        x shape: (# video, 1 frame, y-size, x-size, [HSV channels])
        returns: (# video, y-size, x-size)
        """

        # Accept and convert HSV to grayscale
        if len(x.shape) > 4:
            x = x[..., 2]
        video_n, _, img_y, img_x = x.shape
        predictions = np.zeros((video_n, img_y, img_x), dtype=np.float32)

        # TODO Add frame averaging feature to increase accuracy?
        if len(x) > N_PROCESSES:
            lbp_x = self.lbp_process_frames_multiproc(x[:, 0])
        else:
            lbp_x = self.lbp_process_frames(x[:, 0])

        for i, img in enumerate(x[:, 0]):  # img shape: (y-size, x-size)
            # Reshape LBP image to 3x3 patches (padding if necessary) to be turned into feature vectors for that segment
            lbp_blocks, blk_dims = sliding_win_select(lbp_x[i], self.TEXTURE_BLOCKSIZE)
            # Flatten blocks to a list of feature vectors to be classified by the Random Forest
            lbp_blocks_flat = lbp_blocks.reshape((-1, self.TEXTURE_BLOCKSIZE * self.TEXTURE_BLOCKSIZE))
            # Get classifier predictions
            if prob_mode:
                block_predictions = self.classifier.predict_proba(lbp_blocks_flat)[:, 1].reshape(blk_dims)
            else:
                block_predictions = self.classifier.predict(lbp_blocks_flat).reshape(blk_dims)
            # Augment block predictions to get us to the size of our padded original picture
            mask_img = get_mask_from_blocks(block_predictions, self.TEXTURE_BLOCKSIZE)
            predictions[i] = mask_img[:img.shape[0], :img.shape[1]]  # Cut off the padding to return same sized image

        # TODO Add multithreading

        # Return the list of points (in this case input frames)
        if prob_mode:
            return predictions
        else:
            return predictions > 0.5


class ColorClassifierGMM(Classifier):

    def __init__(self):
        super().__init__('gmm color')
        self.classifier = GaussianMixture()
        self.lowest_prob_pt = None
        self.highest_prob_pt = None

    def train(self, x: np.ndarray, y: np.ndarray, max_pts=1000, visualize=False):
        """
        This procedure will compare whether a low saturation to value ratio is a good indicator of water, scanning
        through the VideoWaterDatabase.

        1. Load the frames of a video with water and non-water in it.
        2. Extract the pixels of the video, flatten then, and group them by water or non-water pixels.
        3. Compute the sat/val ratio for every pixel and output a graph of sat vs. val for every pixel, marking the
           water pixels as blue and non-water pixels as red.
        """

        # Take the first frame from every masked water video and place the HSV values of the water and non-water
        # pixels into a large array. With this array I will be able to develop a MoG model for the most common HSV
        # value combinations for water.

        wat_hsv_data = []
        nowat_hsv_data = []

        x = x[:, 0, :, :]

        for frame, mask in zip(x, y):
            wat_hsv_data.append(frame.reshape((-1, 3))[mask.reshape(-1)])
            # nowat_hsv_data.append(frame.reshape((-1, 3))[~mask.reshape(-1)])
        wat_hsv_data = np.vstack(wat_hsv_data)
        if len(wat_hsv_data) == 0:
            raise RuntimeError('Can\'t train classifier without some water pixels')

        train_pts = np.random.choice(wat_hsv_data.shape[0], size=max_pts, replace=False)
        print('Training gaussian mixture model with %d points' % len(train_pts))
        self.classifier.fit(wat_hsv_data[train_pts])

        # Calculate bounds for scores
        dist = np.zeros(self.classifier.means_.shape, dtype=np.uint8)
        dist[self.classifier.means_ < 127] = 255
        self.lowest_prob_pt = self.classifier.score(dist)
        self.highest_prob_pt = self.classifier.score(self.classifier.means_)

        if visualize:
            fig, axes = plt.subplots(2)
            fig.suptitle(
                'Comparison of the HSV distributions of water/non-water pixels in the VideoWaterDatabase dataset')

            # Water histograms
            bins = 50

            axes[0].set_title('Water Color')
            axes[0].hist(wat_hsv_data[:, 0], range=[0, 255], bins=bins, label='Hue', alpha=0.5, density=True)
            axes[0].hist(wat_hsv_data[:, 1], range=[0, 255], bins=bins, label='Saturation', alpha=0.5, density=True)
            axes[0].hist(wat_hsv_data[:, 2], range=[0, 255], bins=bins, label='Value', alpha=0.5, density=True)
            axes[0].legend()

            # No water histograms
            axes[1].set_title('No Water Color')
            axes[1].hist(nowat_hsv_data[:, 0], range=[0, 255], bins=bins, label='Hue', alpha=0.5, density=True)
            axes[1].hist(nowat_hsv_data[:, 1], range=[0, 255], bins=bins, label='Saturation', alpha=0.5, density=True)
            axes[1].hist(nowat_hsv_data[:, 2], range=[0, 255], bins=bins, label='Value', alpha=0.5, density=True)
            axes[1].legend()

            fig.set_size_inches(15, 15)
            plt.savefig('hsv_comparison_graph', dpi=200)

    def segment(self, x: np.ndarray, prob_mode=False):
        """
        :param x: Shape (# videos, 1 frame, y-size, x-size, HSV channels)
        :param prob_mode: Whether to output probabilities or binary predictions
        :return: Shape (# videos, y-size, x-size)
        """

        videos_n, _, img_y, img_x, _ = x.shape
        predictions = np.zeros((videos_n, img_y, img_x), dtype=np.float32)

        for vid_n, vid_frame in enumerate(x[:, 0]):  # vid_frame Shape: (y-size, x-size, HSV channels)
            log_scores = self.classifier.score_samples(vid_frame.reshape((-1, 3))).reshape(vid_frame.shape[:-1])

            # Normalize scores
            log_scores += abs(self.lowest_prob_pt)
            log_scores[log_scores < 0] = 0
            log_scores /= (self.highest_prob_pt + abs(self.lowest_prob_pt))

            predictions[vid_n] = log_scores if prob_mode else log_scores > 0.5

        return predictions


class TemporalClassifierFFT(Classifier):

    PATCH_BLOCKSIZE = 3
    MAX_TRAINING_PTS = 400000
    MAX_VIDS_BEFORE_THREADING = 5

    def __init__(self, n_frames=60):
        super().__init__('temporal fft', img_classify=False, n_frames=n_frames)
        self.classifier = RandomForestClassifier(n_jobs=N_PROCESSES)
        self.fft_length = None
        self.unit_normalize = False

    def train(self, x: np.ndarray, y: np.ndarray, max_signals=1000, prefer='processes', unit_normalize=True):
        """
        Accepts a 4-dimensional ndarray in the shape of: (# video, # frames, y-size, x-size, [HSV channels])
        :param unit_normalize: Whether to normalize the FFT signals before training
        :param prefer:
        :param x: Input frame sequence
        :param y: Water mask in the shape of (# video, y-size, x-size)
        :param max_signals: Sample this many signals from the frame sequence
        :return: None
        """

        self.unit_normalize = unit_normalize

        if 5 < len(x.shape) < 4:
            raise RuntimeError('Input frame sequence must have 4-5 dimensions')
        if len(x.shape) == 5:
            x = x[..., -1]  # Select V channel in HSV image

        if len(y.shape) != 3:
            raise RuntimeError('Water masks must have 3 dimensions')

        def preprocessing_worker(x_samp, y_samp, normalize) -> np.ndarray:
            dataset = None
            for i, (frame_seq, mask) in enumerate(zip(x_samp, y_samp)):
                print('Training on frame sequence %d/%d' % (i, len(x_samp)))
                # Yields a padded frame sequence of shape: (# frames, # of blocks, blocksize, blocksize)
                pad_frame_seq, padded_frame_dims = sliding_win_select(frame_seq, self.PATCH_BLOCKSIZE)
                pad_mask, padded_mask_dims = sliding_win_select(mask, self.PATCH_BLOCKSIZE)
                # Shape: (# frames, # blocks), bool dtype
                pad_mask = pad_mask.sum(axis=(-1, -2)) > (self.PATCH_BLOCKSIZE * self.PATCH_BLOCKSIZE)//2
                # Signals is an array of the average intensities of 3x3 patches throughout the frame sequence
                # Shape: (# frames, # of blocks or intensity patches)
                signals = pad_frame_seq.mean(axis=(-1, -2))

                # Apply FFT across signals
                fft_signals = np.array([abs(np.fft.fft(signal)) for signal in signals.T]).T
                # fft_signals = (fft_signals.transpose() - fft_signals.mean(axis=1)).transpose()
                # fft_signals = normalize(fft_signals)
                if normalize:
                    fft_signals = ((fft_signals - fft_signals.min(axis=0)) /
                                   (fft_signals.max(axis=0) - fft_signals.min(axis=0))).T
                else:
                    fft_signals = fft_signals.T
                fft_signals[np.isnan(fft_signals)] = 0

                # Split signals into water and non-water by given mask
                wat_signals = fft_signals[pad_mask]
                no_wat_signals = fft_signals[~pad_mask]
                samp_wat_sigs = np.random.choice(len(wat_signals), size=max_signals) \
                    if len(wat_signals) > 0 else []
                samp_no_wat_sigs = np.random.choice(len(no_wat_signals), size=max_signals) \
                    if len(no_wat_signals) > 0 else []

                if dataset is None:
                    dataset = np.hstack((
                        np.vstack((wat_signals[samp_wat_sigs], no_wat_signals[samp_no_wat_sigs])),
                        np.vstack((np.ones((len(samp_wat_sigs), 1)), np.zeros((len(samp_no_wat_sigs), 1))))
                    ))
                else:
                    new_pts = dataset = np.hstack((
                        np.vstack((wat_signals[samp_wat_sigs], no_wat_signals[samp_no_wat_sigs])),
                        np.vstack((np.ones((len(samp_wat_sigs), 1)), np.zeros((len(samp_no_wat_sigs), 1))))
                    ))
                    dataset = np.vstack((dataset, new_pts))
            return dataset

        # Split up workload for multithreading
        x_split = np.array_split(x, N_PROCESSES)
        y_split = np.array_split(y, N_PROCESSES)
        results = joblib.Parallel(n_jobs=N_PROCESSES, prefer=prefer)([joblib.delayed(preprocessing_worker)
                                                                      (x_work, y_work, unit_normalize)
                                                                      for (x_work, y_work) in zip(x_split, y_split)])

        # Compile dataset and sample randomly to reduce training time
        dataset = np.vstack(results)
        if len(dataset) < self.MAX_TRAINING_PTS:
            samp_dataset = dataset
        else:
            samp_dataset = dataset[np.random.choice(len(dataset), size=self.MAX_TRAINING_PTS, replace=False)]

        # Store FFT length for segmentation
        self.fft_length = samp_dataset[:, :-1].shape[-1]

        # Train classifier
        self.classifier.fit(samp_dataset[:, :-1], samp_dataset[:, -1:])

    def segment(self, x: np.ndarray, prob_mode=False, prefer='processes'):
        """
        Accepts an ndarray of a frame sequence in the shape of (# video, # frames, y-size, x-size) and returns a segmentation
        mask from that clip. NOTE: The sequence of frames must be of the same length as the sequences this classifier
        was trained on, i.e. only 60 frames.

        :param prefer:
        :param x: Frame sequence to be identified (# video, # frames, y-size, x-size, [HSV channels])
        :param prob_mode: Flag to return an image of floats that represent the probability of that segment being water
        :return: A sequence of images of shape (# video, y-size, x-size) and dtype of bool or float, depending on prob_mode
        """

        # Convert to HSV if input shape has 5 dimensions
        if len(x.shape) == 5:
            x = x[..., 2]

        # Check trained length of FFT signals to warn about truncation
        if x.shape[1] != self.classifier.n_features_:
            print('[WARNING] Segmentation received a different number of frames (%d) than necessary for FFT segmentation (%d)' %
                  (x.shape[1], self.classifier.n_features_))
            x = x[:, :self.classifier.n_features_]

        if len(x.shape) != 4:
            raise RuntimeError('Cannot accept x array of dimension length %d, only accept 4' % len(x.shape))

        total_vid_n, _, img_y, img_x = x.shape

        def segmentation_worker(x_work: np.ndarray, blocksize: int, normalize: bool):
            segmented_masks = np.zeros((x_work.shape[0], img_y, img_x))
            for i, frame_seq in enumerate(x_work):
                # Yields a padded frame sequence of shape: (# frames, # of blocks, blocksize, blocksize)
                pad_frame_seq, padded_frame_dims = sliding_win_select(frame_seq, self.PATCH_BLOCKSIZE)
                # Signals is an array of the average intensities of 3x3 patches throughout the frame sequence
                # Shape: (# of blocks or intensity patches, # of frames)
                signals = pad_frame_seq.mean(axis=(-1, -2))

                # Apply FFT across signals
                fft_signals = np.array([abs(np.fft.fft(signal)) for signal in signals.T]).T

                # Normalize FFT signals
                if normalize:
                    fft_signals = ((fft_signals - fft_signals.min(axis=0)) /
                                   (fft_signals.max(axis=0) - fft_signals.min(axis=0))).T
                    fft_signals[np.isnan(fft_signals)] = 0
                else:
                    fft_signals = fft_signals.T

                # Train classifier
                probs_signals_blocks = self.classifier.predict_proba(fft_signals)[:, 1].reshape(padded_frame_dims)
                probs_signals_augmented = get_mask_from_blocks(probs_signals_blocks, blocksize)
                probs_signals = probs_signals_augmented[:frame_seq.shape[-2], :frame_seq.shape[-1]]
                if not prob_mode:
                    probs_signals = probs_signals > 0.5
                segmented_masks[i] = probs_signals
            return segmented_masks

        # Intelligent threading - if there are more input videos than the max, then threads are used
        if total_vid_n > self.MAX_VIDS_BEFORE_THREADING:
            x_work = np.array_split(x, N_PROCESSES)
            results = joblib.Parallel(n_jobs=N_PROCESSES, prefer=prefer)(
                [joblib.delayed(segmentation_worker)(x_job, self.PATCH_BLOCKSIZE, self.unit_normalize) for x_job in x_work])
            results = np.vstack(results)
        else:
            results = segmentation_worker(x, self.PATCH_BLOCKSIZE, self.unit_normalize)

        return results


class CombinationClassifier(Classifier):

    MAX_TRAINING_PTS = 1000000

    def __init__(self):
        super(CombinationClassifier, self).__init__('combination', img_classify=False)
        self.classifier = RandomForestClassifier(n_jobs=N_PROCESSES)

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train on probability values output by a certain number of classifiers. Meshes together these maps into one prob
        map.

        :param x: Shape (# vids, y-size, x-size, # classifier outputs) # TODO Change to include frame #
        :param y: Shape (# vids, y-size, x-size)
        :return: None
        """

        pixel_probabilities = x.reshape((-1, x.shape[-1]))
        actual_labels = y.reshape((-1, 1))
        samp = np.random.choice(len(pixel_probabilities), size=self.MAX_TRAINING_PTS, replace=False)
        self.classifier.fit(pixel_probabilities[samp], actual_labels[samp].reshape(-1))

    def segment(self, x: np.ndarray, prob_mode=False):
        # Reshape into a vector of classifier outputs (outputs for each pixel)
        pixel_probabilities = x.reshape((-1, x.shape[-1]))
        # Reshape into x's shape but without the classifier probability dimension
        try:
            return self.classifier.predict_proba(pixel_probabilities)[:, 1].reshape(x.shape[:-1])
        except NotFittedError:
            return None


class TextureTemporalClassifier(Classifier):

    TEXTURE_BLOCKSIZE = 10

    def __init__(self, n_frames=60):
        super(TextureTemporalClassifier, self).__init__('lbp texture')
        self.temporal_classifier = TemporalClassifierFFT(n_frames=n_frames)

    def train(self, x: np.ndarray, y: np.ndarray, max_blocks=100000, prefer='processes', unit_normalize=True):
        """
        Train texture classifier.

        :param prefer:
        :param x: Array of frames (# videos, # frames, y-size, x-size, [HSV channels])
        :param y: Array of masks (# videos, y-size, x-size)
        :param max_blocks: Max 10x10 LBP blocks to train on
        :return:
        """

        print('[TRAIN] Training Texture-Temporal Classifier')

        if len(x.shape) > 4:
            print('Converting 3-channel HSV images to grayscale')
            x = x[..., 2]

        img_y, img_x = x.shape[-2:]
        # lbp_x = TextureClassifierLBP.lbp_process_frames_multiproc(x.reshape((-1, img_y, img_x)), prefer=prefer).reshape(x.shape)
        lbp_x = TextureClassifierLBP.lbp_process_frames_multiproc(x.reshape((-1, img_y, img_x))).reshape(x.shape)
        self.temporal_classifier.train(lbp_x, y, prefer=prefer, unit_normalize=unit_normalize)

    def segment(self, x: np.ndarray, prob_mode=False):
        if len(x.shape) > 4:
            print('Converting 3-channel HSV images to grayscale')
            x = x[..., 2]

        img_y, img_x = x.shape[-2:]
        lbp_x = TextureClassifierLBP.lbp_process_frames_multiproc(x.reshape((-1, img_y, img_x))).reshape(x.shape)
        return self.temporal_classifier.segment(lbp_x, prob_mode=prob_mode)

