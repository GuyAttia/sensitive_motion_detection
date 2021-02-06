import numpy as np
import cv2


def find_interesting_points(image):
    """
    Pass through a Canny edge detector to get a view of all the edges of the object.
    Using Harris corner detector we will find all the interest points of the edge view.
    :param image: Input image as a matrix
    """
    pass


def load_image(image_path):
    """
    Load image from specified path
    :return: Image as a matrix
    """
    pass


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

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path,
                          fourcc,
                          25,
                          (y, x),
                          True)

    for i in range(video.shape[0]):
        out.write(video[i].astype(np.uint8))

    out.release()


def save_bag_of_interesting_points():
    pass


def load_bag_of_interesting_points():
    pass
