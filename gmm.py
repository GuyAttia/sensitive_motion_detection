import warnings
import numpy as np
import pickle
import multiprocessing as mp
from os import path
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm

warnings.filterwarnings('ignore')


class KGmm:
    def __init__(self, k=4, t=0.5, alpha=2.5, learning_rate=0.1, univariate=True):
        """
        Initiate K Multivariate Gaussian distributions for a pixel
        :param k: K Gaussian to work with
        :param t: T threshold for choosing B Gaussian's
        :param alpha: Threshold alpha for deciding if background or foreground (posterior probability to be background)
        :param learning_rate: Learning rate parameter - how much influence the last frame have
        """
        self.k = k
        self.t = t
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.univariate = univariate

        self.mean = np.zeros(k) if self.univariate else np.zeros((k, 3))
        self.cov = np.zeros(k)
        self.weights = np.zeros(k)
        self.b_gaussian = None

    def warm_up(self, first_frames):
        """
        First fit of the GMM using first frames
        :param first_frames: First frames of the video
        """
        # Fit a GMM
        self.weights[0] = 1
        self.mean[0] = first_frames.mean(axis=0)
        self.cov[0] = 11
        self.choose_b_gaussian()

    def choose_b_gaussian(self):
        """
        Choose B dominant gaussian's using the formula to keep minimum weight (T)
        """
        stds = np.sqrt(self.cov)
        gaus_weights = self.weights
        k_tmp = np.divide(gaus_weights, stds)
        k_tmp_dict = {i: v for i, v in enumerate(k_tmp)}
        k_tmp_dict = dict(sorted(k_tmp_dict.items(), key=lambda item: item[1]))

        agg_weight = 0
        self.b_gaussian = []
        for k_index, k_val in k_tmp_dict.items():
            if agg_weight < self.t:
                self.b_gaussian.append(k_index)
                agg_weight += gaus_weights[k_index]
            else:
                return

    def predict_if_foreground(self, frame):
        """
        Predict if the pixel in the frame is foreground or background
        :param frame: New frame to predict on
        :return: Boolean indicator telling if pixel is foreground in the current frame
        """
        # Iterate over the B gaussian and check if one of them is background
        b_gauss_masks = [self.decide_pixel_mask(gauss_index, frame) for gauss_index in self.b_gaussian]
        return max(b_gauss_masks)

    def decide_pixel_mask(self, gauss_index, frame):
        """
        Decide if the pixel value is foreground or background
        :param gauss_index:
        :param frame:
        :return: 1 for foreground and 0 for background
        """
        gauss_mean = self.mean[gauss_index]
        gauss_cov = self.cov[gauss_index]

        if self.univariate:
            diff_ = np.abs(np.subtract(gauss_mean, frame))
        else:
            gauss_std = np.linalg.inv(gauss_cov * np.eye(3))
            pixel_diff = np.abs(np.subtract(gauss_mean, frame))
            diff_ = np.dot(pixel_diff.T, np.dot(gauss_std, pixel_diff))

        if diff_ <= (self.alpha * np.sqrt(gauss_cov)):
            return 0
        else:
            return 1

    def calc_gauss_proba(self, gauss_index, frame):
        """

        :param gauss_index:
        :param frame:
        :return:
        """
        gauss_mean = self.mean[gauss_index]
        gauss_cov = self.cov[gauss_index] if self.cov[gauss_index] != 0 else 1
        if self.univariate:
            frame_proba = norm(loc=gauss_mean, scale=gauss_cov).pdf(frame)
        else:
            gauss_cov = gauss_cov * np.eye(3)
            frame_proba = multivariate_normal(mean=gauss_mean, cov=gauss_cov).pdf(frame)
        return frame_proba

    def update_gmm(self, frame):
        """
        Update the K Multivariate Gaussian using the new frame of the pixel
        :param frame: New frame to update by
        """
        # Choose B dominant gaussian
        self.choose_b_gaussian()

        # Iterate over the B gaussian and check if one of them is background
        m = np.zeros(self.k)    # masking array (used for updating weights)
        predicted_bg = False

        for gauss_index in self.b_gaussian:
            m[gauss_index] = self.decide_pixel_mask(gauss_index=gauss_index, frame=frame)

            # If background - update relevant gaussian
            if m[gauss_index] == 0:
                predicted_bg = True
                frame_proba = self.calc_gauss_proba(gauss_index=gauss_index, frame=frame)
                p = self.learning_rate * frame_proba    # Learning rate
                # Update Cov
                gauss_cov = self.cov[gauss_index]
                if self.univariate:
                    new_cov = ((1 - p) * gauss_cov) + (p * np.square(np.subtract(frame, self.mean[gauss_index])))
                else:
                    new_cov = ((1 - p) * gauss_cov) + (p * np.dot((frame - self.mean[gauss_index]).T,
                                                                  (frame - self.mean[gauss_index])))
                self.cov[gauss_index] = new_cov
                # Update Mean
                gauss_mean = self.mean[gauss_index]
                new_mean = ((1-p) * gauss_mean) + (p * frame)
                self.mean[gauss_index] = new_mean
                break
        if predicted_bg:  # Update weights
            new_weights = ((1 - self.learning_rate) * self.weights) + (self.learning_rate * m)
            new_weights = new_weights / new_weights.sum()
            self.weights = new_weights
            return 0
        else:  # If its foreground - Update the Set of Gaussian
            gauss_probas = [self.calc_gauss_proba(gauss_index=gauss_index, frame=frame)
                            for gauss_index in range(self.k)]
            # Find least probably distribution
            min_proba_index = np.argmin(gauss_probas)
            # Update Mean
            self.mean[min_proba_index] = frame
            # Update Cov
            self.cov[min_proba_index] = 11
            return 1


class GmmModel:
    def __init__(self, k, t, alpha, learning_rate, model_path, univariate):
        """
        Initiate the GMM model using a predefined alpha
        :param k: K Gaussian to work with
        :param t: T threshold for choosing B Gaussian's
        :param alpha: The weight of a new frame on the background updating
        :param learning_rate: Learning rate of a frame
        :param model_path: Path to save the object pickle
        """
        self.k = k
        self.t = t
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.univariate = univariate

        self.update_indices = 0
        self.pixels = None
        self.x = None
        self.y = None

    def generate_gmm_per_pixel(self, frame_shape, first_frames):
        """
        Generate a GMM for each pixel using the first frame
        :param frame_shape: The frame shape
        :param first_frames: First frames of the video to initiate the distributions
        """
        # Iterate over the pixels - create a GMM distribution for each pixel to hold its information
        if self.univariate:
            self.x, self.y = frame_shape
        else:
            self.x, self.y, _ = frame_shape
        self.pixels = [[None] * self.y for _ in range(self.x)]

        for i in tqdm(range(self.x)):
            for j in range(self.y):
                first_frames_pixel = first_frames[:, i, j]
                pixel_model = KGmm(k=self.k, t=self.t, alpha=self.alpha, learning_rate=self.learning_rate,
                                   univariate=self.univariate)
                pixel_model.warm_up(first_frames_pixel)
                self.pixels[i][j] = pixel_model

    @staticmethod
    def update_one_pixel(args):
        single_pixel_obj, frame_pixel = args
        return single_pixel_obj.update_gmm(frame_pixel)

    def apply(self, frame):
        """
        Update the GMM model using new frame and return the frame foreground mask
        :param frame: New frame to update the model by
        """
        pool = mp.Pool(mp.cpu_count()-1)

        with pool:
            args = [(self.pixels[i][j], frame[i, j]) for i in range(self.x) for j in range(self.y)]
            frame_mask = pool.map(self.update_one_pixel, args)

        frame_mask = np.array(frame_mask, dtype=np.uint8).reshape(self.x, self.y)
        return frame_mask

    # def apply_old(self, frame):
    #     """
    #     Update the GMM model using new frame and return the frame foreground mask
    #     :param frame: New frame to update the model by
    #     """
    #     frame_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    #     for i in range(self.x):
    #         for j in range(self.y):
    #             frame_pixel = frame[i, j]
    #             pixel_mask = self.pixels[i][j].update_gmm(frame_pixel)
    #             frame_mask[i, j] = pixel_mask
    #     return frame_mask

    def predict(self, frame):
        """
        """
        frame_mask = np.zeros(frame.shape, dtype=np.uint8)
        for i in range(len(self.pixels)):
            for j in range(len(self.pixels[i])):
                frame_pixel = frame[i, j]
                pixel_mask = self.pixels[i][j].predict_if_foreground(frame_pixel)
                frame_mask[i, j] = pixel_mask
        return frame_mask

    def save_model(self):
        """
        Pickle the model into a file
        """
        with open(self.model_path, 'wb') as f:
            pickle.dump(self, f)


def run_gmm(video, k=4, t=0.5, alpha=2.5, learning_rate=0.01, k_warm_up=20, univariate=True, model_path='model.pickle',
            predict=False):
    # Initiate the GMM class
    if path.isfile(model_path):
        with open(model_path, 'rb') as f:
            gmm_model = pickle.load(f)
            print('GMM Model loaded from a file')
    else:
        gmm_model = GmmModel(k=k, t=t, alpha=alpha, learning_rate=learning_rate, model_path=model_path,
                             univariate=univariate)
        print('GMM model initialized')
        frame_shape = video[0].shape
        warm_up_frames = video[:k_warm_up]
        print('Start generating GMM object for each pixel on warm up frames')
        gmm_model.generate_gmm_per_pixel(frame_shape, warm_up_frames)
        print('Done generating GMM object for each pixel')

    fg_mask_list = []
    print('Start iterating over frames to detect foreground')
    for frame_index, frame in enumerate(tqdm(video)):
        if frame_index < k_warm_up:
            fg_mask = np.zeros(frame.shape[:2])
        else:
            if predict:
                fg_mask = gmm_model.predict(frame)
            else:
                fg_mask = gmm_model.apply(frame)
        fg_mask_list.append(fg_mask)
    v_fg_mask = np.array(fg_mask_list)

    gmm_model.save_model()
    print('GMM Model saved to a file')
    return v_fg_mask
