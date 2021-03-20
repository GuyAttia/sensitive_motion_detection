import cv2
import numpy as np


def improve_foreground(v_fg_mask):
    """
    Use postprocessing functions like erosion and dilation to improve the foreground mask
    :param v_fg_mask: Foreground mask
    :return: Improved foreground mask
    """
    v_fg_mask_pp = np.zeros(v_fg_mask.shape, dtype=np.uint8)

    # Iterate over the video
    for frame_index, frame in enumerate(v_fg_mask):
        # Erosion and dilation on the binary mask
        frame_pp = cv2.erode(frame, np.ones((5, 5), np.uint8))
        frame_pp = cv2.erode(frame_pp, np.ones((3, 3), np.uint8))
        frame_pp = cv2.dilate(frame_pp, np.ones((5, 5), np.uint8))
        frame_pp = cv2.dilate(frame_pp, np.ones((12, 12), np.uint8))

        v_fg_mask_pp[frame_index] = frame_pp

    return v_fg_mask_pp


def find_contours(v_fg_mask):
    """
    If detected a significant change, frame the contours of the objects using a rectangle and return the two
    objects with the largest area
    :param v_fg_mask: A mask video which we detected a significant change
    :return: Two largest contours if existed in every frame
    """
    frames_contours = {}
    for frame_index, frame in enumerate(v_fg_mask):
        contours, _, = cv2.findContours(frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the largest area
        if contours:
            frames_contours[frame_index] = {}
            two_largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            c1 = two_largest_contours[0]
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            frames_contours[frame_index]['object1'] = [x1, y1, w1, h1]

            if len(two_largest_contours) > 1:
                c2 = two_largest_contours[1]
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), 255, 1)
                frames_contours[frame_index]['object2'] = [x2, y2, w2, h2]
    return frames_contours
