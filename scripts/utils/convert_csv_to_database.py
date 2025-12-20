import json
import os
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv

import db_manager.db_manager as db_m


def find_unique_images(matches_df: pd.DataFrame):
    image_id_set: set = set()

    unique_patches_1 = matches_df["file1"].unique()
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
        db_m.add_image_by_id(image)


def add_patches_to_db(base_dir: str, image_id_set: set):
    for image_id in image_id_set:
        patch_dir_path = os.path.join(base_dir, image_id)
        json_path = os.path.join(patch_dir_path, f"{image_id}_patch_info.json")

        with open(json_path, "r") as file:
            data = json.load(file)

        for key in data.keys():
            patch_data = data[key]
            filename = patch_data["filename"]
            patch_id = filename.split(".")[0]
            coords = patch_data["coordinates"]
            [left, top, right, bottom] = coords

            db_m.add_patch(patch_id, image_id, filename, left, top, right, bottom)


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


def add_matches_to_db(matches_df: pd.DataFrame):
    for idx, row in matches_df.iterrows():
        patch_id_1 = row["file1"].split(".")[0]
        patch_id_2 = row["file2"].split(".")[0]
        if patch_id_2 < patch_id_1:
            patch_id_1, patch_id_2 = patch_id_2, patch_id_1

        distance = row["distance"]
        sum_homo_err = row["sum_homo_err"]
        len_homo_err = row["len_homo_err"]
        mean_homo_err = row["mean_homo_err"]
        std_homo_err = row["std_homo_err"]

        db_m.add_match(
            patch_id_1,
            patch_id_2,
            distance,
            sum_homo_err,
            len_homo_err,
            mean_homo_err,
            std_homo_err,
        )

        if (idx + 1) % 10000 == 0:
            print(f"match {idx + 1}")


def main():
    args = load_args()

    print(args.matches_df.shape)
    # images = find_unique_images(args.matches_df)

    # add_images_to_db(images)
    # add_patches_to_db(args.patches_path, images)
    # db_m.add_entire_df_to_db(args.matches_df)


if __name__ == "__main__":
    main()
