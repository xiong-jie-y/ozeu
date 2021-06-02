import glob
import json
import os
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

    u2_mask_model = U2MaskModel(U2MaskModelOption(model_name=model_name))
    annotations_train = []
    annotations_val = []
    images_train = []
    images_val = []

    mask_val_path = os.path.join(output_path, "masks")
    os.makedirs(mask_val_path, exist_ok=True)
    image_val_path = os.path.join(output_path, "images")
    os.makedirs(image_val_path, exist_ok=True)

    mask_path = os.path.join(output_path, "masks")
    os.makedirs(mask_path, exist_ok=True)
    image_output_path = os.path.join(output_path, "images")
    os.makedirs(image_output_path, exist_ok=True)

    # hand_segmenter = SegnetHandSegmenter()
    hand_segmenter = InstanceSegmenter()

    # refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

    next_image_id = 0
    for dataset in datasets:
        with tempfile.TemporaryDirectory() as image_path:
            video_path = dataset['video_path']
            file_template = os.path.join(image_path, "$filename%06d.png")
            command = f"ffmpeg -i {join(dataset_definition_file_dir, video_path)} -r {fps}/1 {file_template}"
            subprocess.run(command, shell=True, check=True)

            image_paths = sorted(list(glob.glob(join(image_path, "*.png"))) + list(glob.glob(join(image_path, "*.jpg"))))
            areas = []
            masks = []
            annotations = []
            new_image_names = []
            images = []
            image_data = []
            AREA_THREASHOLD = 10000
            for image_path in tqdm.tqdm(image_paths):
                image_bgr = cv2.imread(image_path)
                image_bgr = cv2.resize(image_bgr, (image_bgr.shape[1] // resize_factor, image_bgr.shape[0] // resize_factor))
                mask = u2_mask_model.predict_mask(image_bgr)
                image_bgr_for_display = image_bgr.copy()
                mask_numpy = mask.squeeze(0).cpu().detach().numpy()
                mask_u8 = u2_mask_model.convert_probability_mask_to_binary_mask(mask_numpy)

                if remove_hand and dataset["category_id"] not in hand_class_ids:
                    hand_mask_u8 = hand_segmenter.get_mask(image_bgr, class_names=["hand"])
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

                image_name = os.path.basename(image_path)
                video_name = os.path.basename(video_path)
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
                    "category_id": dataset["category_id"],
                    "id": next_image_id
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

            WINDOW_LENGTH = 5
            ALLOWABLE_MEDIAN_DEVIATION = 0.40
            # area_moving_median = np.convolve(areas, np.ones(WINDOW_LENGTH), 'valid') / WINDOW_LENGTH
            # area_moving_median = scipy.signal.medfilt(areas, WINDOW_LENGTH)
            effective_window = deque([])
            # st.write(area_moving_median)
            near_median_indices = []
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

            # import IPython; IPython.embed()

            # fig, ax = plt.subplots(1, 1)
            # ax.plot(list(range(0, len(areas))), areas)
            # st.pyplot(fig)

            # import IPython; IPython.embed()

            NUM_VAL_PER_NUM = 5
            num_added = 0

            for i in near_median_indices:
                if num_added % NUM_VAL_PER_NUM == 0:
                    annotations_val.append(annotations[i])
                    images_val.append(images[i])
                    cv2.imwrite(os.path.join(mask_val_path, new_image_names[i]), masks[i]) 
                    # cv2.imwrite(os.path.join(image_output_path, os.path.splitext(new_image_names[i])[0] + ".jpg"), image_data[i])
                    cv2.imwrite(os.path.join(image_val_path, new_image_names[i]), image_data[i])

                else:
                    annotations_train.append(annotations[i])
                    images_train.append(images[i])

                    cv2.imwrite(os.path.join(mask_path, new_image_names[i]), masks[i]) 
                    # cv2.imwrite(os.path.join(image_output_path, os.path.splitext(new_image_names[i])[0] + ".jpg"), image_data[i])
                    cv2.imwrite(os.path.join(image_output_path, new_image_names[i]), image_data[i])

                num_added += 1

    json.dump({
        "images": images_train,
        "annotations": annotations_train,
        "categories": categories,
        "info": {
            "description": "COCO 2017 Dataset"
        }
    }, open(os.path.join(output_path, "annotations.json"), "w"), indent=4)

    json.dump({
        "images": images_val,
        "annotations": annotations_val,
        "categories": categories,
        "info": {
            "description": "COCO 2017 Dataset"
        }
    }, open(os.path.join(output_path, "annotations_val.json"), "w"), indent=4)

if __name__ == "__main__":
    main()
