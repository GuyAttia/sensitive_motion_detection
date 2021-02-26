# from utils import find_interesting_points, load_bag_of_interesting_points
import numpy as np


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

    
def generate_patch_histogram(patch, window_size):
    """
    Calculate histogram of a given matrix
    :return: Normalized histogram of the given matrix
    """
    pixel_counts,_ = np.histogram(patch, range(257))
    pixel_counts = pixel_counts / window_size**2
    return pixel_counts

def calculate_histograms_mean(hist_list, num_of_top_pixels):
    """
    Calculate the mean of requested number of top pixels
    :return: The mean pixel value of 'num_of_top_pixels' pixels in of the histogram
    """
    patch_hist_mean = []
    for h in hist_list:
        patch_hist_mean.append(h.argsort()[:num_of_top_pixels].mean())
    return sum(patch_hist_mean) / len(patch_hist_mean)

def generate_patches_histograms_mean(image_list, window_size, num_of_top_pixels = 50):
    """
    Create a list of mean pixel value for images list
    :return: A list containing for each image in the list the mean 
             pixel value of the top 'num_of_top_pixels' pixels in the image
    """
    histograms_mean_list = []
    for image in image_list: 
        histograms_list = []
        for x in range(window_size, image.shape[0], window_size):
            for y in range(window_size, image.shape[1], window_size):
                patch_histogram = generate_patch_histogram(image[x-window_size:x+window_size, y-window_size:y+window_size], window_size)
                histograms_list.append(patch_histogram)
        histograms_mean_list.append(calculate_histograms_mean(histograms_list, num_of_top_pixels))
    return histograms_mean_list

def calculate_allowed_histograms_means(histograms_mean_list, bins):
    """
    Create a list of allowed mean histogram values will be used to identify new objects
    :return: A list of given 'bins' length containing the mean values of histograms
    """
    allowed_hist_values = []
    pixel_counts,indices = np.histogram(histograms_mean_list, bins)
    print(indices)
    for idx in range(0,len(indices)-1):
        cluster_values = [val for val in histograms_mean_list if val>=indices[idx] and val<=indices[idx+1]]
        if len(cluster_values) > 0:
            allowed_hist_values.append(sum(cluster_values)/len(cluster_values))
    return allowed_hist_values