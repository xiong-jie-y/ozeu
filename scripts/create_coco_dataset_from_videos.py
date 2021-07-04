import glob
import json
import os
from ozeu.utils.coco import CocoDatasetBuilder
import subprocess
import tempfile
from collections import deque
from os.path import join

import click
import cv2
import numpy as np
import tqdm
import yaml
from ozeu.detection.instance_segmentation import InstanceSegmenter
from ozeu.segmentation.u2net_wrapper import U2MaskModel, U2MaskModelOption
from ozeu.utils.binary_image import create_roi_from_u8_mask

# import segmentation_refinement as refine

class SalientObjectAnnotator:
    def __init__(self, model_name, dataset_builder, resize_factor, remove_hand, hand_class_ids, dataset, input_is_sequential_frame):
        self.u2_mask_model = U2MaskModel(U2MaskModelOption(model_name=model_name))
        self.hand_segmenter = InstanceSegmenter()
        self.dataset_builder = dataset_builder
        self.resize_factor = resize_factor
        self.remove_hand = remove_hand
        self.hand_class_ids = hand_class_ids
        self.dataset = dataset
        self.input_is_sequential_frame = input_is_sequential_frame

    # hand_segmenter = SegnetHandSegmenter()

    # refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

    def process(self, image_paths, video_name):
        areas = []
        masks = []
        annotations = []
        new_image_names = []
        images = []
        image_data = []
        AREA_THREASHOLD = 10000
        for image_path in tqdm.tqdm(image_paths):
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Error {image_path}")
                continue

            image_bgr = cv2.resize(image_bgr, (image_bgr.shape[1] // self.resize_factor, image_bgr.shape[0] // self.resize_factor))
            AREA_THREASHOLD = image_bgr.shape[1] * image_bgr.shape[0] * 0.05
            mask = self.u2_mask_model.predict_mask(image_bgr)
            image_bgr_for_display = image_bgr.copy()
            mask_numpy = mask.squeeze(0).cpu().detach().numpy()
            mask_u8 = self.u2_mask_model.convert_probability_mask_to_binary_mask(mask_numpy)

            if self.remove_hand and self.dataset["category_id"] not in self.hand_class_ids:
                hand_mask_u8 = self.hand_segmenter.get_mask(image_bgr, class_names=["hand"]).get_positive_black_image()
                hand_mask_u8 = cv2.erode(hand_mask_u8, kernel=np.ones((3,3)), iterations=2)
                mask_u8[hand_mask_u8 == 0] = 0
                hand_mask_color = np.zeros(image_bgr.shape, dtype=np.uint8)
                hand_mask_color[hand_mask_u8 == 0, :] = [0, 255, 0]
                image_bgr_for_display = cv2.addWeighted(image_bgr_for_display, 1.0, hand_mask_color, 0.8, 0)
                cv2.imshow("test2", hand_mask_u8)

            mask_u8 = cv2.erode(mask_u8, kernel=np.ones((3,3)), iterations=4)
            mask_u8 = cv2.dilate(mask_u8, kernel=np.ones((3,3)), iterations=4)

            original_mask = np.zeros(image_bgr.shape, dtype=np.uint8)
            original_mask[mask_u8 == 255, :] = [0, 0, 255]
            # mask_u8 = refiner.refine(image_bgr, mask_u8, fast=False, L=750) 
            # refined_mask = np.zeros(image_bgr.shape, dtype=np.uint8)
            # refined_mask[mask_u8 > 1, :] = [0, 0, 255]

            # image_bgr = cv2.addWeighted(image_bgr, 1.0, refined_mask, 0.3, 0)
            image_bgr_for_display = cv2.addWeighted(image_bgr_for_display, 0.5, original_mask, 0.5, 0)
            cv2.imshow("Original Image", image_bgr_for_display)

            roi = create_roi_from_u8_mask(mask_u8)
            if roi is None:
                continue
            contour_points = roi.contours[0].flatten().tolist()
            area = cv2.contourArea(np.array(roi.contours[0])) if len(roi.contours) != 0 else 0

            # masks.append(mask_u8)
            cv2.imshow("test", mask_u8)
            cv2.waitKey(1)

            if area < AREA_THREASHOLD:
                continue

            next_image_id = self.dataset_builder.get_next_image_id()
            image_name = os.path.basename(image_path)
            new_image_name = f"{next_image_id}_{video_name}_{image_name}"
            new_image_names.append(new_image_name)

            areas.append(area)

            # import IPython; IPython.embed()
            annotation = {
                "segmentation": [contour_points],
                # "area": float(max_x - min_x) * float(max_y - min_y),
                "area": area,
                "iscrowd": 0,
                "image_id": next_image_id,
                "bbox": [
                    float(roi.min_x), float(roi.min_y), 
                    float(roi.max_x - roi.min_x), float(roi.max_y - roi.min_y)],
                "category_id": self.dataset["category_id"],
                "id": self.dataset_builder.get_next_annotation_id()
            }
            image = {
                "file_name": new_image_name,
                "width": image_bgr.shape[1],
                "height": image_bgr.shape[0],
                "id": next_image_id,
                "image_id": next_image_id
            }
            annotations.append(annotation)
            images.append(image)

            # shutil.copy2(image_path, os.path.join(image_output_path, new_image_name))
            image_data.append(image_bgr)
            masks.append(mask_u8)
            next_image_id += 1

        assert len(areas) == len(annotations)
        assert len(annotations) == len(image_data)
        assert len(annotations) == len(masks)

        near_median_indices = []
        if self.input_is_sequential_frame:
            WINDOW_LENGTH = 5
            ALLOWABLE_MEDIAN_DEVIATION = 0.40
            # area_moving_median = np.convolve(areas, np.ones(WINDOW_LENGTH), 'valid') / WINDOW_LENGTH
            # area_moving_median = scipy.signal.medfilt(areas, WINDOW_LENGTH)
            effective_window = deque([])
            # st.write(area_moving_median)
            for i, area in enumerate(areas):
                area_moving_median = None
                if len(effective_window) > 0:
                    area_moving_median = np.median(effective_window)
                    # area_moving_median = np.mean(effective_window)

                if area_moving_median is None or (area_moving_median * (1-ALLOWABLE_MEDIAN_DEVIATION) < area and area < area_moving_median * (1+ALLOWABLE_MEDIAN_DEVIATION)):
                    if len(effective_window) == WINDOW_LENGTH:
                        effective_window.popleft()
                    effective_window.append(areas[i])

                    near_median_indices.append(i)
        else:
            near_median_indices = list(range(0, len(areas)))

        # import IPython; IPython.embed()

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(list(range(0, len(areas))), areas)
        # st.pyplot(fig)

        # import IPython; IPython.embed()

        NUM_VAL_PER_NUM = 5
        num_added = 0

        for i in near_median_indices:
            if num_added % NUM_VAL_PER_NUM == 0:
                self.dataset_builder.add_data([annotations[i]], images[i], new_image_names[i], masks[i], image_data[i], 'val')
            else:
                self.dataset_builder.add_data([annotations[i]], images[i], new_image_names[i], masks[i], image_data[i], 'train')

            num_added += 1

@click.command()
@click.option("--model-name")
@click.option("--dataset-definition-file")
@click.option("--output-path")
@click.option("--hand-class-ids", type=int, multiple=True)
@click.option("--remove-hand", is_flag=True)
@click.option("--fps", type=float)
@click.option("--resize-factor", type=int, default=1)
def main(model_name, dataset_definition_file, hand_class_ids, output_path, fps, resize_factor, remove_hand):
    dataset_definition_file_dir = os.path.dirname(dataset_definition_file)
    dataset_definition = yaml.load(open(dataset_definition_file, "r"))
    datasets = dataset_definition['datasets']
    categories = dataset_definition['categories']

    dataset_builder = CocoDatasetBuilder(output_path)
    dataset_builder.add_categories(categories)

    for dataset in datasets:
        with tempfile.TemporaryDirectory() as image_path:
            video_path = dataset['video_path']
            file_path = join(dataset_definition_file_dir, video_path)
            process_video = os.path.isfile(file_path)
            if process_video:
                file_template = os.path.join(image_path, "$filename%06d.png")
                command = f"ffmpeg -i {file_path} -r {fps}/1 {file_template}"
                subprocess.run(command, shell=True, check=True)
            else:
                image_path = file_path

            annotator = SalientObjectAnnotator(model_name, dataset_builder, resize_factor, remove_hand, hand_class_ids, dataset, process_video)

            image_paths = sorted(list(glob.glob(join(image_path, "*.png"))) + list(glob.glob(join(image_path, "*.jpg"))))
            print(f"Processing {image_path}")
            annotator.process(image_paths, os.path.basename(video_path))

    dataset_builder.finalize_dataset()

if __name__ == "__main__":
    main()
