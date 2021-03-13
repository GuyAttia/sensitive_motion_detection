# from utils import find_interesting_points, load_bag_of_interesting_points
import numpy as np


def match_with_approved_objects(num_of_objects, objects_info, approved_object_info):
    """
    NEED TO MODIFY
    """
    approved_objects = []
    for obj_idx in range(num_of_objects):
        red = objects_info['RED'][obj_idx]
        green = objects_info['GREEN'][obj_idx]
        blue = objects_info['BLUE'][obj_idx]
        location = objects_info['location'][obj_idx]
        for approved_obj_idx in range(len(approved_object_info['RED'])):
            if ((red <= (approved_object_info["RED"][approved_obj_idx] + 10) and red >= (approved_object_info["RED"][approved_obj_idx] - 10)) and \
                (green <= (approved_object_info["GREEN"][approved_obj_idx] + 10) and red >= (approved_object_info["GREEN"][approved_obj_idx] - 10)) and \
                (blue <= (approved_object_info["BLUE"][approved_obj_idx] + 10) and red >= (approved_object_info["BLUE"][approved_obj_idx] - 10))):
                if (location['grid_size'] <= (approved_object_info["location"][0]['grid_size'] + 35) and \
                    location['grid_size'] >= (approved_object_info["location"][0]['grid_size'] - 35)):
                    approved_objects.append(obj_idx)
    return approved_objects
                    
    
def get_detected_objects_info(image, detected_objects, image_x_grid, image_y_grid, window_size):
    '''
    NEED TO MODIFY
    '''
    objects_info = {"RED":[], "GREEN":[], "BLUE":[], "location":[]}
    for obj in detected_objects:
        location = detected_objects[obj]
        cropped_object = np.array(image[location[2]:location[3], location[0]:location[1]])
        
        for ch_num, channel in enumerate(objects_info):
            if ch_num == 3: break
            objects_info[channel].append(generate_patches_histograms_mean(cropped_object[:,:,ch_num], window_size))
            
        location_info = get_grids(image_x_grid, image_y_grid, location)
        objects_info["location"].append(location_info)
        
    return objects_info
                            
    
def generate_patch_histogram(patch, window_size):
    """
    Calculate histogram of a given matrix
    :return: Normalized histogram of the given matrix
    """
    pixel_counts,_ = np.histogram(patch, range(257))
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

def generate_patches_histograms_mean(image, window_size, num_of_top_pixels = 50):
    """
    Create a list of mean pixel value for images list
    :return: A list containing for each image in the list the mean 
             pixel value of the top 'num_of_top_pixels' pixels in the image
    """
    histograms_list = []
    for x in range(window_size, image.shape[0], window_size):
        for y in range(window_size, image.shape[1], window_size):
            patch_histogram = generate_patch_histogram(image[x-window_size:x+window_size, y-window_size:y+window_size], window_size)
            histograms_list.append(patch_histogram)
    histograms_mean = calculate_histograms_mean(histograms_list, num_of_top_pixels)
    return histograms_mean

def calculate_allowed_histograms_means(histograms_mean_dict, bins):
    """
    Create a list of allowed mean histogram values will be used to identify new objects
    :return: A list of given 'bins' length containing the mean values of histograms
    """
    allowed_hist_values = {"RED":[], "GREEN":[], "BLUE":[]}
    for channel in histograms_mean_dict:
        pixel_counts,indices = np.histogram(histograms_mean_dict[channel], bins)
        for idx in range(0,len(indices)-1):
            cluster_values = [val for val in histograms_mean_dict[channel] if val>=indices[idx] and val<=indices[idx+1]]
            if len(cluster_values) > 0:
                allowed_hist_values[channel].append(sum(cluster_values)/len(cluster_values))
    return allowed_hist_values

def get_grids(image_x_grid, image_y_grid, object_location):
    """
    Calculate the object location in the image grid.
    :return: A dictionary containing the middle of the object in he grid coordinates. 
             The number of grid points the object is spread on.
             The index of all grid points the object is spread on.
    """
    object_grid_details = {}
    
    object_x_grid = np.where((image_x_grid>object_location[0]) & (image_x_grid<object_location[1]))
    object_y_grid = np.where((image_y_grid>object_location[2]) & (image_y_grid<object_location[3]))
    
    object_grid_details['middle_grid'] = [np.mean(object_x_grid), np.mean(object_y_grid)]
    object_grid_details['grid_size'] = object_x_grid[0].shape[0] * object_y_grid[0].shape[0]
    object_grid_details['grid_coordinates'] = {'X': object_x_grid, "Y": object_y_grid}
    
    return object_grid_details