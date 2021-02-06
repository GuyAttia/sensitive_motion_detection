from utils import find_interesting_points, save_bag_of_interesting_points


def find_interesting_points_in_pictures(images_list):
    """
    Find interesting points for each picture
    :param images_list: List of images
    """
    for image in images_list:
        find_interesting_points(image=image)


def find_matching_points(images_dict):
    """
    Find matching points between all of the interesting points:
    For each image we will calculate it’s matching points with the other images.
    Points who will be found as matching in a determined number of images will stay,
    and points who don't have a matching in the determined number of images will be deleted.

    :param images_list: List of images and their interesting points
    """
    pass
