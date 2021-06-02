from typing import List
from pkg_resources import resource_filename
from ozeu.utils.file import get_model_file, get_model_file_from_gdrive
import numpy as np

from mmdet.apis import inference_detector, init_detector

import cv2

class InstanceSegmenter:
    def __init__(self):
        device = 'cuda:0'
        checkpoint_file = get_model_file_from_gdrive("hand_instance_segmentation_v0_1.pth", "https://drive.google.com/u/0/uc?id=1FkH1FFKWUNxzHJNiBFfQLZRzKppijy3K&export=download")
        config_file = resource_filename("ozeu", "detection/configs/yolact/yolact_r50_1x8_coco.py")
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.previous_seg = None

    def get_mask(self, color_image_bgr: np.ndarray, class_names: List[str]):
        result = inference_detector(self.model, color_image_bgr)

        detection_image = self.model.show_result(
            color_image_bgr, result, score_thr=0.50, wait_time=0, show=False)
        cv2.imshow("Debug: Detected Objects", detection_image)
        segmentations = result[1]

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result[0])
        ]

        mask = np.zeros(
            (color_image_bgr.shape[0], color_image_bgr.shape[1]), dtype=np.uint8)
        class_indices = [
            self.model.CLASSES.index(class_name)
            for class_name in class_names
        ]

        mask[:, :] = 255
        not_found = True
        for seg_list, label_list, detection_list in zip(segmentations, labels, result[0]):
            for seg, label, detection in zip(seg_list, label_list, detection_list):
                if detection[4] > 0.5 and label in class_indices:
                    mask[seg == 1] = 0
                    self.previous_seg = seg
                    not_found = False

        # if not_found:
        #     mask[self.previous_seg == 1] = 0

        return mask
