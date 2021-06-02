import glob
import json
import os
from os.path import join
import shutil
import os
from PIL import Image, ImageChops, ImageMath
import random
import click
import imageio
import numpy as np


@click.command()
@click.option("--input-dataset-path", required=True)
@click.option("--destination-root", required=True)
@click.option("--augmentation-mode", required=True)
def main(input_dataset_path, destination_root, augmentation_mode):

    output_dirs = [
        input_dataset_path,
    ]

    def get_all_files(directory):
        files = []

        for f in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, f)):
                files.append(os.path.join(directory, f))
            else:
                files.extend(get_all_files(os.path.join(directory, f)))
        return files

    def change_background(img, mask, bg):
        # oh = img.height  
        # ow = img.width
        ow, oh = img.size
        bg = bg.resize((ow, oh)).convert('RGB')

        # ratio = np.random.random() * 0.5 + 0.5
        # resized_width = img.size[0] * ratio
        # resized_height = img.size[1] * ratio
        # shift_x = np.random.randint(0, ow, resized_width)
        # shift_y = np.random.randint(0, oh, resized_height)

        # resized_img = Image.new('RGB', img.size, (255, 255, 255))
        # resized_img_orig = img.resize((resized_width, resized_height))
        # resized_img.paste(resized_img_orig)

        # resized_mask = Image.new('GRAY', img.size, 255)
        # resized_mask_orig = mask.resize((resized_width, resized_height))
        # resized_mask.paste(resized_mask_orig)
        
        imcs = list(img.split())
        bgcs = list(bg.split())
        maskcs = list(mask.split())
        fics = list(Image.new(img.mode, img.size).split())
        
        for c in range(len(imcs)):
            negmask = maskcs[c].point(lambda i: 1 - i / 255)
            posmask = maskcs[c].point(lambda i: i / 255)
            fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
        out = Image.merge(img.mode, tuple(fics))

        return out

    def augment_weekpoint(img):
        import imgaug.augmenters as iaa
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug = iaa.Sequential([
            sometimes(iaa.MotionBlur(k=list(range(7, 25)), angle=[-45, -15, 0, 15, 45])), 
            sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))),
            sometimes(iaa.LogContrast(gain=(0.6, 1.4))),
        ])
        return aug(image=img)

    def augment_weekpoint(img):
        import imgaug.augmenters as iaa
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug = iaa.Sequential([
            sometimes(iaa.MotionBlur(k=list(range(3, 25)), angle=[-45, -15, 0, 15, 45])), 
            sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))),
            sometimes(iaa.imgcorruptlike.GaussianNoise(severity=[1,2])),
            sometimes(iaa.imgcorruptlike.GaussianBlur(severity=[1,2])),
            sometimes(iaa.imgcorruptlike.DefocusBlur(severity=[1,2])),
            iaa.Sometimes(0.1, iaa.imgcorruptlike.ShotNoise(severity=[1])),
            iaa.Sometimes(0.1, iaa.imgcorruptlike.ImpulseNoise(severity=[1])),
            iaa.Sometimes(0.1, iaa.imgcorruptlike.SpeckleNoise(severity=[1])),
            # sometimes(iaa.LogContrast(gain=(0.6, 1.4))),  
        ])
        return aug(image=img)

    os.makedirs(join(destination_root, "images"), exist_ok=True)

    bg_file_names = get_all_files('backgrounds_for_augmentation')

    for dataset_id, output_dir in enumerate(output_dirs):
        # len_frame = len(list(glob.glob(os.path.join(output_dir, "images/*.png"))))

        ext = "png"
        # ext = "jpg"
        # parents = "images/"
        parents = "images/"
        # for i in range(0, len_frame):
        for image_path in glob.glob(os.path.join(output_dir, f"{parents}*.{ext}")):
            i = os.path.splitext(os.path.basename(image_path))[0]
            random_bg_index = random.randint(0, len(bg_file_names) - 1)

            if augmentation_mode == "different_background":
                image = Image.open(os.path.join(output_dir, f"{parents}{i}.{ext}")).convert('RGB')
                mask =  Image.open(os.path.join(output_dir, f"masks/{i}.png")).convert('RGB')
                background =  Image.open(bg_file_names[random_bg_index]).convert('RGB')
                aug_img = change_background(image, mask, background)
                aug_img.save(os.path.join(destination_root, f"{parents}{i}.{ext}"))
            elif augmentation_mode == "weekpoint":
                image = imageio.imread(os.path.join(output_dir, f"{parents}{i}.{ext}"))
                aug_img = augment_weekpoint(image)
                imageio.imwrite(os.path.join(destination_root, f"{parents}{i}.{ext}"), aug_img)
            else:
                raise RuntimeError(f"No such augment type {augmentation_mode}.")

        shutil.copy(
            join(output_dir, "annotations.json"),
            join(destination_root, "annotations.json")
        )
        shutil.copy(
            join(output_dir, "annotations_val.json"),
            join(destination_root, "annotations_val.json")
        )

if __name__ == "__main__":
    main()