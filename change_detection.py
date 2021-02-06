import numpy as np


def change_dection(video, k1, k2, background_th):
    """
    Decide if there was a significant change in the video or not.
    """
    v_foreground = np.zeros(video.shape)

    # Run over the video
    for frame_index in range(video.shape[0]):
        if frame_index % k2 == 0:  # Update median
            first_index = 0 if frame_index - k1 < 0 else frame_index - k1
            previous_frames = video[first_index:frame_index, :, :, :]
            i_b_0 = np.median(previous_frames[:, :, :, 0], axis=0)
            i_b_1 = np.median(previous_frames[:, :, :, 1], axis=0)
            i_b_2 = np.median(previous_frames[:, :, :, 2], axis=0)

        # Current frame RGB
        frame_0 = video[frame_index, :, :, 0]
        frame_1 = video[frame_index, :, :, 1]
        frame_2 = video[frame_index, :, :, 2]

        # Compute point-wise difference for each color metric
        d_frame_0 = np.abs(np.subtract(i_b_0, video[frame_index, :, :, 0]))
        d_frame_1 = np.abs(np.subtract(i_b_1, video[frame_index, :, :, 1]))
        d_frame_2 = np.abs(np.subtract(i_b_2, video[frame_index, :, :, 2]))

        # Decide if foreground or not for each color metric
        decision_0 = d_frame_0 > background_th
        decision_1 = d_frame_1 > background_th
        decision_2 = d_frame_2 > background_th

        # Decide for the entire pixel
        decision_all = np.multiply(decision_0, decision_1, decision_2)

        # Set the intensity for the pixel in the new video (black or the actual color from the video)
        v_frame_0 = np.multiply(frame_0, decision_all)
        v_frame_1 = np.multiply(frame_1, decision_all)
        v_frame_2 = np.multiply(frame_2, decision_all)
        v_frame = np.dstack((v_frame_0, v_frame_1, v_frame_2)).astype(np.uint8)
        v_foreground[frame_index] = v_frame

    return v_foreground


def choose_best_frame(video):
    """
    If detected a significant change, choose the frame that includes most of the object
    :param video: A video which we detected a significant change
    :return: The best frame for our object
    """
    pass


def prepare_object(image):
    """
    Fill gaps and crop the object from the best frame as a preparation for the object classification
    :param image: The frame that includes most of the object
    :return: Filled cropped object as an image
    """
    pass
