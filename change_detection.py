import cv2
import numpy as np

from gmm import run_gmm


def change_detection(video, k, t, alpha, learning_rate, k_warm_up=20, univariate=True,
                     model_path='pycharm_model.pickle', predict=False):
    """
    Decide if there was a significant change in the video or not.
    Combine both algorithms (GMM and Clustering) by giving weights to each one and decide using their
    combined decision whether there was a change and where. The result of the algorithms will be a binary mask.
    """
    v_fg_mask = run_gmm(video, k=k, t=t, alpha=alpha, learning_rate=learning_rate, k_warm_up=k_warm_up,
                        univariate=univariate, model_path=model_path, predict=predict)
    return v_fg_mask


def openCV_change_detection(video):
    # Todo: Remove before submission
    fgbg = cv2.createBackgroundSubtractorMOG2()
    v_fg_mask = np.zeros(video.shape[:3])

    for frame_index in range(video.shape[0]):
        frame = video[frame_index]
        fg_mask = fgbg.apply(frame)
        v_fg_mask[frame_index] = fg_mask

    return v_fg_mask


def improve_foreground(v_fg_mask):
    """
    Use postprocessing functions like erosion and dilation to improve the foreground mask
    :param v_fg_mask: Foreground mask
    :return: Improved foreground mask
    """
    v_fg_mask_pp = np.zeros(v_fg_mask.shape, dtype=np.uint8)

    # Iterate over the video
    for frame_index, frame in enumerate(v_fg_mask):
        # Erosion and dilation on the binary mask
        frame_pp = cv2.erode(frame, np.ones((5, 5), np.uint8))
        frame_pp = cv2.erode(frame_pp, np.ones((3, 3), np.uint8))
        frame_pp = cv2.dilate(frame_pp, np.ones((5, 5), np.uint8))
        frame_pp = cv2.dilate(frame_pp, np.ones((12, 12), np.uint8))

        v_fg_mask_pp[frame_index] = frame_pp

    return v_fg_mask_pp


def recreate_video(orig_vid, v_fg_mask):
    """
    """
    v_fg = np.zeros(orig_vid.shape, dtype=np.uint8)

    # Iterate over the video
    for frame_index, (orig_frame, fg_mask_frame) in enumerate(zip(orig_vid, v_fg_mask)):
        # Set the intensity for the pixel in the new video (black or the actual color from the video)
        fg_mask_frame = np.repeat(fg_mask_frame[:, :, np.newaxis], 3, axis=2)
        v_pp_frame = np.multiply(orig_frame, fg_mask_frame)
        v_fg[frame_index] = v_pp_frame

    return v_fg


def find_contours(v_fg_mask):
    """
    If detected a significant change, frame the contours of the objects using a rectangle and return the two
    objects with the largest area
    :param v_fg_mask: A mask video which we detected a significant change
    :return: Two largest contours if existed in every frame
    """
    frames_contours = {}
    for frame_index, frame in enumerate(v_fg_mask):
        contours, _ = cv2.findContours(frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the largest area
        if contours:
            frames_contours[frame_index] = {}
            two_largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            c1 = two_largest_contours[0]
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            frames_contours[frame_index]['object1'] = [x1, y1, w1, h1]

            if len(two_largest_contours) > 1:
                c2 = two_largest_contours[1]
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), 255, 1)
                frames_contours[frame_index]['object2'] = [x2, y2, w2, h2]
    return frames_contours


if __name__ == '__main__':
    from os import path, listdir
    from utils import load_video, resize, save_video

    univariate = True
    cut_percent = 100
    predict = True
    model_path = path.join('data', 'models', f'{univariate}-{cut_percent}.pickle')
    videos_dir_path = path.join('data', 'videos')
    videos_list = [video_name for video_name in listdir(videos_dir_path) if video_name.endswith('mp4')]

    for video_name in videos_list:
        print(f'Change detection start for video {video_name}')
        video_path = path.join('data', 'videos', f'{video_name}')
        video_fg_path = path.join('data', 'foreground', f'Predict-Pycharm-{univariate}-{cut_percent}-{video_name}')
        video_mask_path = path.join('data', 'masks', f'Pycharm-{univariate}-{cut_percent}-{video_name}')
        video_mask_pp_path = path.join('data', 'masks', f'Pycharm-{univariate}-{cut_percent}_pp-{video_name}')
        if path.isfile(video_fg_path) and path.isfile(video_mask_path):
            continue
        colored_vid = load_video(video_path=video_path, gray_scale=False)
        colored_vid = resize(colored_vid, percent=cut_percent, gray_scale=False)

        if univariate:
            vid = load_video(video_path=video_path, gray_scale=True)
            vid = resize(vid, percent=cut_percent, gray_scale=True)
        else:
            vid = colored_vid.copy()

        v_fg_mask = change_detection(vid, k=4, t=0.7, alpha=2.5, learning_rate=0.05, k_warm_up=1, univariate=univariate,
                                     model_path=model_path, predict=predict)
        # save_video(v_fg_mask, video_mask_path, color=False)
        v_fg_mask_pp = improve_foreground(v_fg_mask)
        # save_video(v_fg_mask_pp, video_mask_pp_path, color=False)
        frames_contours = find_contours(v_fg_mask_pp)

        v_fg = recreate_video(orig_vid=colored_vid, v_fg_mask=v_fg_mask_pp)
        save_video(v_fg, video_fg_path)
        print(f'Change detection done for video {video_name}')
        break