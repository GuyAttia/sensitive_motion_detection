import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def find_interesting_points(image):
    """
    Pass through a Canny edge detector to get a view of all the edges of the object.
    Using Harris corner detector we will find all the interest points of the edge view.
    :param image: Input image as a matrix
    """
    pass


def load_images(images_path, cmap = cv2.IMREAD_COLOR):
    """
    Load images from specified path
    :return: Images as a matrix
    """
    image_list = [cv2.imread(img, cmap) for img in glob.glob(f'{images_path}*.jpg')]
    return image_list
    
    
def save_images_with_boxes(img, idx, approved_objects, detected_objects_location, out_dir = 'data/object_classification/'):
    image_with_rectangle = img
    for i in range(len(detected_objects_location)):
        if i in approved_objects:#:f"object{i+1}" in detected_objects[0].keys():
            image_with_rectangle = cv2.rectangle(image_with_rectangle, 
                                                 (detected_objects_location[f'object{i+1}'][0],detected_objects_location[f'object{i+1}'][2]), 
                                                 (detected_objects_location[f'object{i+1}'][1],detected_objects_location[f'object{i+1}'][3]), 
                                                 (255,0,0), 2)
        else:
            image_with_rectangle = cv2.rectangle(image_with_rectangle, 
                                                 (detected_objects_location[f'object{i+1}'][0],detected_objects_location[f'object{i+1}'][2]), 
                                                 (detected_objects_location[f'object{i+1}'][1],detected_objects_location[f'object{i+1}'][3]), 
                                                 (0,0,255), 2)
    cv2.imwrite(f'{out_dir}/detected{idx}.jpg', image_with_rectangle)    

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
