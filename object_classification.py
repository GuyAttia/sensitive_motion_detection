import numpy as np


def match_with_approved_objects(num_of_objects, objects_info, approved_object_info):
    """
    Compare the approved objects information with the current frame calculated objects info.
    :return: Indices list of the approved objects.
    """
    approved_objects = []
    not_approved_objects = []
    for obj_idx in range(num_of_objects):
        red = objects_info['RED'][obj_idx]
        green = objects_info['GREEN'][obj_idx]
        blue = objects_info['BLUE'][obj_idx]
        location = objects_info['location'][obj_idx]
        for approved_obj in approved_object_info:
            for approved_obj_idx in range(len(approved_obj['RED'])):
                if ((red <= (approved_obj["RED"][approved_obj_idx] + 20) and red >= (
                        approved_obj["RED"][approved_obj_idx] - 20)) and \
                        (green <= (approved_obj["GREEN"][approved_obj_idx] + 20) and red >= (
                                approved_obj["GREEN"][approved_obj_idx] - 20)) and \
                        (blue <= (approved_obj["BLUE"][approved_obj_idx] + 20) and red >= (
                                approved_obj["BLUE"][approved_obj_idx] - 20)) and \
                        (location['grid_size'] <= (approved_obj["location"][0]['grid_size'] + 500) and \
                         location['grid_size'] >= (approved_obj["location"][0]['grid_size'] - 500))):
                    if not obj_idx in approved_objects: approved_objects.append(obj_idx)
                elif approved_obj["location"][0]['grid_size'] / 5 < location['grid_size']:
                    if not obj_idx in not_approved_objects: not_approved_objects.append(obj_idx)
    return approved_objects, not_approved_objects


def get_detected_objects_info(image, detected_objects, image_x_grid, image_y_grid, window_size):
    '''
    Generate the given frame objects information based on the input image and detected objects location.
    :return: A dictionary describing the mean RGB values for each object and it's location in the image.
    '''
    objects_info = {"RED": [], "GREEN": [], "BLUE": [], "location": []}
    for obj in detected_objects:
        location = detected_objects[obj]
        cropped_object = np.array(image[location[1]:location[1] + location[3], location[0]:location[0] + location[2]])
        for ch_num, channel in enumerate(objects_info):
            if ch_num == 3: break
            if window_size > location[2] or window_size > location[3]:
                objects_info[channel].append(0)
            elif image.shape == (location[3], location[2], 3):
                objects_info[channel].append(0)
            else:
                objects_info[channel].append(
                    generate_patches_histograms_mean(cropped_object[:, :, ch_num], window_size))

        location_info = get_grids(image_x_grid, image_y_grid, location)
        objects_info["location"].append(location_info)

    return objects_info


def generate_patch_histogram(patch, window_size):
    """
    Calculate histogram of a given matrix
    :return: Normalized histogram of the given matrix
    """
    pixel_counts, _ = np.histogram(patch, range(257))
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


def generate_patches_histograms_mean(image, window_size, num_of_top_pixels=50):
    """
    Create a list of mean pixel value for images list
    :return: A list containing for each image in the list the mean 
             pixel value of the top 'num_of_top_pixels' pixels in the image
    """
    histograms_list = []
    for x in range(window_size, image.shape[0], window_size):
        for y in range(window_size, image.shape[1], window_size):
            patch_histogram = generate_patch_histogram(
                image[x - window_size:x + window_size, y - window_size:y + window_size], window_size)
            histograms_list.append(patch_histogram)
    histograms_mean = calculate_histograms_mean(histograms_list, num_of_top_pixels)
    return histograms_mean


def generate_histograms_mean_dict(image_list, window_size):
    histograms_mean_dict = {"RED": [], "GREEN": [], "BLUE": []}
    for img_num, img in enumerate(image_list):
        r_histograms_mean_list = generate_patches_histograms_mean(img[:, :, 0], window_size)
        g_histograms_mean_list = generate_patches_histograms_mean(img[:, :, 1], window_size)
        b_histograms_mean_list = generate_patches_histograms_mean(img[:, :, 2], window_size)
        histograms_mean_dict["RED"].append(r_histograms_mean_list)
        histograms_mean_dict["GREEN"].append(g_histograms_mean_list)
        histograms_mean_dict["BLUE"].append(b_histograms_mean_list)
    return histograms_mean_dict


def calculate_allowed_histograms_means(histograms_mean_dict, bins):
    """
    Create a list of allowed mean histogram values will be used to identify new objects
    :return: A list of given 'bins' length containing the mean values of histograms
    """
    allowed_hist_values = {"RED": [], "GREEN": [], "BLUE": []}
    for channel in histograms_mean_dict:
        pixel_counts, indices = np.histogram(histograms_mean_dict[channel], bins)
        for idx in range(0, len(indices) - 1):
            cluster_values = [val for val in histograms_mean_dict[channel] if
                              val >= indices[idx] and val <= indices[idx + 1]]
            if len(cluster_values) > 0:
                allowed_hist_values[channel].append(sum(cluster_values) / len(cluster_values))
    return allowed_hist_values


def get_grids(image_x_grid, image_y_grid, object_location):
    """
    Calculate the object location in the image grid.
    :return: A dictionary containing the middle of the object in he grid coordinates. 
             The number of grid points the object is spread on.
             The index of all grid points the object is spread on.
    """
    object_grid_details = {}

    object_x_grid = np.where(
        (image_x_grid > object_location[0]) & (image_x_grid < (object_location[0] + object_location[2])))
    object_y_grid = np.where(
        (image_y_grid > object_location[1]) & (image_y_grid < (object_location[1] + object_location[3])))

    object_grid_details['middle_grid'] = [np.mean(object_x_grid), np.mean(object_y_grid)]
    object_grid_details['grid_size'] = object_x_grid[0].shape[0] * object_y_grid[0].shape[0]
    object_grid_details['grid_coordinates'] = {'X': object_x_grid, "Y": object_y_grid}

    return object_grid_details
