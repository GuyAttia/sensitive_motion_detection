from os import path, listdir
from utils import *
from change_detection import *
from object_classification import *


def run(approved_objects):
    univariate = True
    cut_percent = 100
    predict = True
    model_path = path.join('data', 'models', f'17032021{univariate}-{cut_percent}.pickle')
    videos_dir_path = path.join('data', 'new_videos')
    videos_list = [video_name for video_name in listdir(videos_dir_path) if video_name.endswith('mp4')]

    tp = []
    tn = []
    fp = []
    fn = []
    for video_name in videos_list:
        label_ = 0 if 'notalarm' in video_name else 1
        print(f'Change detection start for video {video_name}')
        video_path = path.join('data', 'new_videos', f'{video_name}')
        colored_vid = load_video(video_path=video_path, gray_scale=False)
        # colored_vid = resize(colored_vid, percent=cut_percent, gray_scale=False)

        if univariate:
            vid = load_video(video_path=video_path, gray_scale=True)
            # vid = resize(vid, percent=cut_percent, gray_scale=True)
        else:
            vid = colored_vid.copy()

        v_fg_mask = change_detection(vid, k=4, t=0.7, alpha=2.5, learning_rate=0.1, k_warm_up=1, univariate=univariate,
                                     model_path=model_path, predict=predict)
        v_fg_mask_pp = improve_foreground(v_fg_mask)
        frames_contours = find_contours(v_fg_mask_pp)
        # Todo: Add classification
        prediction_ = 1

        if (label_ == 1) and (prediction_ == 1):
            tp.append(video_name)
        elif (label_ == 0) and (prediction_ == 0):
            tn.append(video_name)
        elif (label_ == 1) and (prediction_ == 0):
            fn.append(video_name)
        elif (label_ == 0) and (prediction_ == 1):
            fp.append(video_name)

        print(f'Change detection done for video {video_name}')

    print(f'TP: {len(tp)}')
    print(f'TN: {len(tn)}')
    print(f'FN: {len(fn)}')
    print(f'FP: {len(fp)}')

    accuracy = (len(tp) + len(tn)) / len(videos_list)
    recall = len(tp) / (len(tp) + len(fn))
    precision = len(tp) / (len(tp) + len(fp))
    # print(f'Accuracy: {accuracy}')
    # print(f'Recall: {recall}')
    # print(f'Precision: {precision}')

    return videos_list, tp, tn, fn, fp, accuracy, recall, precision