from collections import defaultdict
import json
import os

import cv2

SPLITS_ = ['train', 'val']

class CocoDatasetBuilder:
    def __init__(self, output_path):
        self.next_image_id = 0
        self.next_annotation_id = 0

        self.output_path = output_path

        self.annotations = defaultdict(list)
        self.images = defaultdict(list)

        self.mask_path = os.path.join(output_path, "masks")
        os.makedirs(self.mask_path, exist_ok=True)
        self.image_output_path = os.path.join(output_path, "images")
        os.makedirs(self.image_output_path, exist_ok=True)

    def add_categories(self, categories):
        self.categories = categories

    def get_next_image_id(self):
        self.next_image_id += 1
        return self.next_image_id

    def get_next_annotation_id(self):
        self.next_annotation_id += 1
        return self.next_annotation_id

    def add_data(self, annotations, image, image_name, mask, image_data, split):
        self.annotations[split] += annotations
        self.images[split].append(image)
        cv2.imwrite(os.path.join(self.mask_path, image_name), mask) 
        # cv2.imwrite(os.path.join(image_output_path, os.path.splitext(new_image_names[i])[0] + ".jpg"), image_data[i])
        cv2.imwrite(os.path.join(self.image_output_path, image_name), image_data)

    def finalize_dataset(self):
        for split in SPLITS_:
            json.dump({
                "images": self.images[split],
                "annotations": self.annotations[split],
                "categories": self.categories,
                "info": {
                    "description": "COCO 2017 Dataset"
                }
            }, open(os.path.join(self.output_path, "annotations.json"), "w"), indent=4)

    # def add_image(self, image_name, image_bgr, split):
    #     self.images[split].append({
    #         "file_name": image_name,
    #         "width": image_bgr.shape[1],
    #         "height": image_bgr.shape[0],
    #         "id": self.next_image_id,
    #         "image_id": self.next_image_id #maybe not necessary
    #     })
    #     self.next_image_id += 1

    #     return self.next_image_id

    # def add_annotation(self, roi, contour_points, area, image_id, category_id):
    #     annotation = {
    #         "segmentation": [contour_points],
    #         # "area": float(max_x - min_x) * float(max_y - min_y),
    #         "area": area,
    #         "iscrowd": 0,
    #         "image_id": image_id,
    #         "bbox": [
    #             float(roi.min_x), float(roi.min_y), 
    #             float(roi.max_x - roi.min_x), float(roi.max_y - roi.min_y)],
    #         "category_id": category_id,
    #         "id": self.next_anotation_id
    #     }
    #     self.next_anotation_id += 1