import json
import os
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv

import db_manager.db_manager as db_m


def find_unique_images(matches_df: pd.DataFrame):
    image_id_set: set = set()

    print("start_big_op_1")
    unique_patches_1 = matches_df["file1"].unique()
    print("start_big_op_2")
    unique_patches_2 = matches_df["file2"].unique()

    patch_set_1: set = set(unique_patches_1)
    patch_set_2: set = set(unique_patches_2)
    patch_set = patch_set_1.union(patch_set_2)

    for patch in patch_set:
        patch_id = patch.split(".")[0]
        img_id = patch_id.split("_")[0]
        image_id_set.add(img_id)

    return image_id_set


def add_images_to_db(image_id_set: set):
    for image in image_id_set:
        db_m.add_image_by_name(image)


def add_patches_to_db(base_dir, image_id_set: set):
    for image_id in image_id_set:
        patch_dir_path = os.path.join(base_dir, image_id)
        json_path = os.path.join(patch_dir_path, f"{image_id}_patch_info.json")

        with open(json_path, "r") as file:
            data = json.load(file)

        print(data)


def load_args():
    load_dotenv()

    args = {}  # Use a dictionary to store environment variables
    args["base_path"] = os.getenv("BASE_PATH")
    args["output_base"] = os.getenv("OUTPUT_BASE_PATH")

    args["image_list"] = os.getenv("PAM_FILES_TO_PROCESS")
    if args["image_list"] is not None:
        args["image_list"] = args["image_list"].split(",")

    args["image_path"] = os.path.join(args["base_path"], os.getenv("IMAGES_IN"))
    args["patches_path"] = os.path.join(args["base_path"], os.getenv("PATCHES_IN"))
    args["csv_file"] = os.path.join(
        args["base_path"], os.getenv("CLEAN_SIFT_MATCHES_W_TP_W_HOMO")
    )

    args["matches_df"] = pd.read_csv(args["csv_file"], low_memory=False)
    args = SimpleNamespace(**args)
    return args


def main():
    args = load_args()

    images = find_unique_images(args.matches_df)

    add_patches_to_db(args.patches_path, images)


if __name__ == "__main__":
    main()
