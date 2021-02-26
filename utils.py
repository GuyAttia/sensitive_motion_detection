import numpy as np
import cv2
import matplotlib.pyplot as plt


def find_interesting_points(image):
    """
    Pass through a Canny edge detector to get a view of all the edges of the object.
    Using Harris corner detector we will find all the interest points of the edge view.
    :param image: Input image as a matrix
    """
    pass


def load_image(image_path, cmap = cv2.IMREAD_GRAYSCALE):
    """
    Load image from specified path
    :return: Image as a matrix
    """
    return cv2.imread(image_path, cmap)
    


def save_image(image, output_path):
    """
    Load image from specified path
    :param image: Image to save
    :param output_path: Full path to save the image in (including image name)
    """
    pass


def load_video(video_path):
    """
    Load video from specified path
    :return: Video as a list of images
    """
    vid = cv2.VideoCapture(video_path)

    images = []
    # Check if camera opened successfully
    if not vid.isOpened():
        print("Error opening video stream or file")
        return np.array(images)

    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            images.append(frame.astype(np.int16))
        else:
            return np.array(images)


def save_video(video, output_path):
    """
    Save a video
    :param video: Video to save as a list of images
    :param output_path: Full path to save the video in (including video name)
    """
    x = video.shape[1]
    y = video.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path,
                          fourcc,
                          25,
                          (y, x),
                          True)

    for i in range(video.shape[0]):
        out.write(video[i].astype(np.uint8))

    out.release()


def generate_patch_histogram(patch, window_size):
#     print(patch)
    pixel_counts,_ = np.histogram(patch, range(257))
    pixel_counts = pixel_counts / window_size**2
    return pixel_counts

def calculate_histograms_mean(hist_list, num_of_top_pixels = 50):
    patch_hist_mean = []
#     num_of_patches = len(hist_list)
    for h in hist_list:
        patch_hist_mean.append(h.argsort()[:num_of_top_pixels].mean())
    return sum(patch_hist_mean) / len(patch_hist_mean)

def save_bag_of_interesting_points():
    pass


def load_bag_of_interesting_points():
    pass


def play_video_by_images(video, frame_rate=20):
    """
    Plot frames of the video
    :param video: Video to play
    :param frame_rate: Rate of frames to show
    """
    for frame_index in range(video.shape[0]):
        if frame_index % frame_rate == 0:
            fig, ax = plt.subplots(1, figsize=(15, 15))
            ax.imshow(video[frame_index])
            ax.set_title(f'Frame {frame_index}')
            plt.show()
