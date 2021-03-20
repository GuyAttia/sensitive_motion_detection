import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os import listdir, path


def load_images(images_path, cmap=cv2.IMREAD_COLOR):
    """
    Load images from specified path
    :return: Images as a matrix
    """
    image_list = [cv2.imread(path.join(images_path, img), cmap) for img in listdir(images_path)]
    return image_list


def save_images_with_boxes(img, idx, approved_objects, not_approved_objects, detected_objects_location,
                           out_dir='data/object_classification/'):
    """
    Set the rectangles in the videos. Set red rectangle for unauthorized objects and blue for authorized
    """
    image_with_rectangle = img
    for i in range(len(detected_objects_location)):
        if i in approved_objects:
            image_with_rectangle = cv2.rectangle(image_with_rectangle,
                                                 (detected_objects_location[f'object{i + 1}'][0],
                                                  detected_objects_location[f'object{i + 1}'][1]),
                                                 (detected_objects_location[f'object{i + 1}'][0] +
                                                  detected_objects_location[f'object{i + 1}'][2],
                                                  detected_objects_location[f'object{i + 1}'][1] +
                                                  detected_objects_location[f'object{i + 1}'][3]),
                                                 (255, 0, 0), 2)
        elif i in not_approved_objects:
            image_with_rectangle = cv2.rectangle(image_with_rectangle,
                                                 (detected_objects_location[f'object{i + 1}'][0],
                                                  detected_objects_location[f'object{i + 1}'][1]),
                                                 (detected_objects_location[f'object{i + 1}'][0] +
                                                  detected_objects_location[f'object{i + 1}'][2],
                                                  detected_objects_location[f'object{i + 1}'][1] +
                                                  detected_objects_location[f'object{i + 1}'][3]),
                                                 (0, 0, 255), 2)
    cv2.imwrite(f'{out_dir}/detected{idx}.jpg', image_with_rectangle)


def load_video(video_path, gray_scale=True):
    """
    Load video from specified path
    :param gray_scale: Load as RGB or GrayScale
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
            if gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images.append(frame.astype(np.uint8))
        else:
            return np.array(images)


def save_video(video, output_path, color=True):
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
                          color)

    for i in range(video.shape[0]):
        out.write(video[i].astype(np.uint8))

    out.release()


def save_frames_video(image_folder, out_video_path):
    """
    Save the frames of the video
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def play_video_by_images(video, frame_rate=20):
    """
    Plot frames of the video
    :param video: Video to play
    :param frame_rate: Rate of frames to show
    """
    for frame_index in range(video.shape[0]):
        if frame_index % frame_rate == 0:
            fig, ax = plt.subplots(1, figsize=(15, 15))
            ax.imshow(video[frame_index].astype(np.uint8))
            ax.set_title(f'Frame {frame_index}')
            plt.show()


def resize(video, percent=50, gray_scale=True):
    """
    Changing video resolution
    :param video: Video to resize
    :param percent: Percent of the video to keep
    :param gray_scale: If the video is in RGB or GrayScale
    :return: Same video in different resolution
    """
    first_frame = video[0]
    width = int(first_frame.shape[1] * percent / 100)
    height = int(first_frame.shape[0] * percent / 100)
    if gray_scale:
        resized_video = np.zeros((video.shape[0], height, width), dtype=np.uint8)
    else:
        resized_video = np.zeros((video.shape[0], height, width, video.shape[3]), dtype=np.uint8)

    for frame_index, frame in enumerate(video):
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, fx=255, fy=255, interpolation=cv2.INTER_CUBIC)
        resized_video[frame_index] = resized_frame

    return resized_video
