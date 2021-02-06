from utils import find_interesting_points, load_bag_of_interesting_points


def match_with_approved_objects(image):
    """
    - Find the image interesting points.
    - Load the bag of interesting points calculated ahead
    - Match the interesting points with the bag of interesting points
    - Decide on alarm or not

    :param image: Detected image with the object
    """
    pass
    find_interesting_points(image=image)
    load_bag_of_interesting_points()
