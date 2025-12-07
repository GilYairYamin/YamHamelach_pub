import json
import os
from argparse import ArgumentParser
from types import SimpleNamespace

from tqdm import tqdm

from env_arguments_loader import load_env_arguments


def load_args():
    """
    Load and parse command line arguments combined with environment settings.

    Loads default values from environment configuration and allows command line
    arguments to override them. Combines both sources into a single namespace.

    Returns:
        SimpleNamespace: Object containing all configuration parameters with
                        attributes for im_path, patches_dir, bbox_dir, cp, and model_type

    Command Line Arguments:
        --im_path: Path to input images directory
        --patches_dir: Path to save extracted patch images
        --bbox_dir: Path to save detection visualization images
        --cp: Path to model checkpoint file
        --model_type: Type of model ("yolo", "faster_rcnn", or "auto")
        --confidence_threshold: Minimum confidence score for detections

    Example:
        >>> args = load_args()
        >>> args.im_path
        '/path/to/input/images'
        >>> args.cp
        '/path/to/model.pt'
        >>> args.model_type
        'auto'
    """
    args = load_env_arguments()

    parser = ArgumentParser()
    parser.add_argument(
        "--im_path",
        help="paths to input images containing multiple patches",
        default=args.images_in,
    )
    parser.add_argument(
        "--patches_dir",
        help="path to save images with bounding boxes and patches crops",
        default=args.patches_dir,
    )
    parser.add_argument(
        "--bbox_dir",
        help="path to save bounding box images",
        default=args.bbox_dir,
    )

    parsed_args = parser.parse_args()
    # return SimpleNamespace(**args, **parsed_args.__dict__)
    merged_dict = {**args.__dict__, **parsed_args.__dict__}
    combined_args = SimpleNamespace(**merged_dict)
    return combined_args


def main():
    args = load_args()

    patches_dir = os.path.join(args.base_path, "new-training-patches-2.12.2025")

    image_dirs = os.listdir(patches_dir)

    result = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [{"id": 1, "name": "patch", "supercategory": ""}],
        "images": [],
        "annotations": [],
    }

    result_images = result["images"]
    result_annotations = result["annotations"]

    for idx, image_dir in tqdm(enumerate(image_dirs, 1)):
        image_name = os.path.basename(image_dir)
        coco_filepath = os.path.join(
            patches_dir, image_dir, f"{image_name}_patch_info_coco.json"
        )
        with open(coco_filepath, mode="r+") as coco_file:
            coco_dict = json.load(coco_file)

        image = coco_dict["images"][0]
        image["id"] = idx
        result_images.append(image)

        annotations = coco_dict["annotations"]
        for annotation in annotations:
            annotation["image_id"] = idx
            result_annotations.append(annotation)

    res_file_path = os.path.join(patches_dir, "coco_annotations.json")
    with open(res_file_path, "w") as file:
        json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
