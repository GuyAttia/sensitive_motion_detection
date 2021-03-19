from os import path, listdir
from utils import *
from change_detection import *
from object_classification import *

def match_with_approved_objects_2(num_of_objects, objects_info, approved_object_info):
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
                if ((red <= (approved_obj["RED"][approved_obj_idx] + 20) and red >= (approved_obj["RED"][approved_obj_idx] - 20)) and \
                    (green <= (approved_obj["GREEN"][approved_obj_idx] + 20) and red >= (approved_obj["GREEN"][approved_obj_idx] - 20)) and \
                    (blue <= (approved_obj["BLUE"][approved_obj_idx] + 20) and red >= (approved_obj["BLUE"][approved_obj_idx] - 20)) and \
                    (location['grid_size'] <= (approved_obj["location"][0]['grid_size'] + 500) and \
                     location['grid_size'] >= (approved_obj["location"][0]['grid_size']  - 500))):
                    if not obj_idx in approved_objects: approved_objects.append(obj_idx)
                elif approved_obj["location"][0]['grid_size'] / 5 < location['grid_size']:
                    if not obj_idx in not_approved_objects: not_approved_objects.append(obj_idx)
#                 elif approved_obj["location"][0]['grid_size'] / 5 < location['grid_size']:
#                     if not obj_idx in not_approved_objects: not_approved_objects.append(obj_idx)
    return approved_objects, not_approved_objects
def run(approved_objects, image_x_grid, image_y_grid, window_size):
    univariate = True
    cut_percent = 100
    predict = True
    model_path = path.join('data', 'models', 'True-100.pickle')
    videos_dir_path = path.join('data', 'videos', 'cut videos')
    videos_list = [video_name for video_name in listdir(videos_dir_path) if video_name.endswith('mp4')]

    tp = []
    tn = []
    fp = []
    fn = []
    for video_name in videos_list:
        label_ = 0 if 'not_alarm' in video_name else 1
        print(f'Change detection start for video {video_name}')
        video_path = path.join('data', 'videos', 'cut videos', f'{video_name}')
        colored_vid = load_video(video_path=video_path, gray_scale=False)

        if univariate:
            vid = load_video(video_path=video_path, gray_scale=True)
            vid = resize(vid, percent=cut_percent, gray_scale=True)
        else:
            vid = colored_vid.copy()

        v_fg_mask = change_detection(vid, k=3, t=0.7, alpha=2.5, learning_rate=0.5, k_warm_up=1, univariate=univariate,
                                     model_path=model_path, predict=predict)
        v_fg_mask_pp = improve_foreground(v_fg_mask)
        frames_contours = find_contours(v_fg_mask_pp)
        print(frames_contours)
        approved_objects_total = 0
        not_approved_objects_total = 0
        for frame_index, detected_objects_location in frames_contours.items():
            frame = colored_vid[frame_index]
            num_of_objects = len(detected_objects_location)
            objects_info = get_detected_objects_info(frame, detected_objects_location, image_x_grid, image_y_grid, window_size)
            print(objects_info)
            approved_objects, not_approved_objects = match_with_approved_objects_2(num_of_objects, objects_info, approved_objects)
            approved_objects_total += len(approved_objects)
            not_approved_objects_total += len(not_approved_objects)
        
        print(approved_objects_total, not_approved_objects_total)
        prediction_ = 0 if approved_objects_total > 5  else 1
        
        if prediction_ == 1: print(f"{video_name} contains moving unauthorized objects")
        if prediction_ == 0: print(f"{video_name} contains moving authorized objects")
            
        if (label_ == 1) and (prediction_ == 1):
            tn.append(video_name)
        elif (label_ == 0) and (prediction_ == 0):
            tp.append(video_name)
        elif (label_ == 0) and (prediction_ == 1):
            fn.append(video_name)
        elif (label_ == 1) and (prediction_ == 0):
            fp.append(video_name)

        print(f'Change detection done for video {video_name}')

    print(f'TP: {len(tp)}')
    print(f'TN: {len(tn)}')
    print(f'FN: {len(fn)}')
    print(f'FP: {len(fp)}')

    accuracy = (len(tp) + len(tn)) / len(videos_list)
    recall = len(tp) / (len(tp) + len(fn))
    precision = len(tp) / (len(tp) + len(fp))

    return videos_list, tp, tn, fn, fp, accuracy, recall, precision