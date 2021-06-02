from ozeu.types.geometry import RectROI
import cv2
import numpy as np


def create_roi_from_u8_mask(u8_mask, margin=0):
    contours, hierarchy = cv2.findContours(u8_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    xs = []
    ys = []
    for contour in contours:
        for pt_cont in contour:
            point = pt_cont[0]
            xs.append(point[0])
            ys.append(point[1])
    
    if len(xs) == 0:
        return None

    min_x = max(np.min(xs) - margin, 0)
    min_y = max(np.min(ys) - margin, 0)
    max_x = min(np.max(xs) + margin, u8_mask.shape[1])
    max_y = min(np.max(ys) + margin, u8_mask.shape[0])

    box = RectROI(min_x, min_y, max_x, max_y, contours)

    return box