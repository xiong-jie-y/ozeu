import glob
import json
import os
import shutil
import click

@click.command()
@click.option("--input-dirs", multiple=True)
@click.option("--destination-root")
def main(input_dirs, destination_root):
    if os.path.exists(destination_root):
        shutil.rmtree(destination_root)

    os.makedirs(os.path.join(destination_root, "images"))

    class AnnotationMerger:
        def __init__(self):
            self.new_annotation_id = 0
            self.image_id_mapper = {}
            self.image_id = 0

            self.annotations = []
            self.images = []
            self.categories = []

        def add_annotations(self, annotation_path):
            annotations_json = json.load(open(annotation_path))
            
            for image in annotations_json["images"]:
                image["file_name"] = "{}_{}".format(dataset_id, image["file_name"])
                self.image_id_mapper[image["id"]] = self.image_id
                image["id"] = self.image_id
                image["image_id"] = self.image_id

                self.image_id += 1
                self.images.append(image)

            for annotation in annotations_json["annotations"]:
                annotation["image_id"] = self.image_id_mapper[annotation["image_id"]]
                annotation["id"] = self.new_annotation_id

                self.new_annotation_id += 1
                self.annotations.append(annotation)

            self.categories = annotations_json['categories']

        def dump(self, path):
            json.dump({
                "annotations": self.annotations,
                "images": self.images,
                "categories": self.categories
            }, open(path, "w"))
            

    annotation_merger = AnnotationMerger()
    annotation_val_merger = AnnotationMerger()
    ext = "png"

   #  print(input_dirs)

    for dataset_id, output_dir in enumerate(input_dirs):
        print(f"processing {output_dir}")
        # len_frame = len(list(glob.glob(os.path.join(output_dir, f"images/*.{ext}"))))

        for image_path in glob.glob(os.path.join(output_dir, f"images/*.{ext}")):
            i = os.path.splitext(os.path.basename(image_path))[0]
            shutil.copy(
                os.path.join(output_dir, f"images/{i}.{ext}"),
                os.path.join(destination_root, f"images/{dataset_id}_{i}.{ext}")
            )

        annotation_merger.add_annotations(os.path.join(output_dir, "annotations.json"))
        annotation_val_merger.add_annotations(os.path.join(output_dir, "annotations_val.json"))

    annotation_merger.dump(os.path.join(destination_root, "annotations.json"))
    annotation_val_merger.dump(os.path.join(destination_root, "annotations_val.json"))

    print(f"dataset output in {destination_root}")

if __name__ == "__main__":
    main()