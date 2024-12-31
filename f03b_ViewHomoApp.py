import ast
import json
import os
import pickle
# import tempfile
# from typing import Tuple

import cv2
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D


# Function to load an image and convert to RGB for display
def load_image(image_path: str):
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(f"Could not load image: {image_path}")
        return None


def load_keypoints(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)
    return data["keypoints"]


# Function to get patch information from a JSON file
def get_patch_info(base_path: str, file_name: str, box: str):
    json_file = os.path.join(base_path, file_name, f"{file_name}_patch_info.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            patch_info = json.load(f)
        return patch_info.get(box)
    return None


# Function to visualize a match between two patches
def visualize_match(row, base_path, image_path, patches_key_dec_cache, debug=False):
    file1, file2 = row["file1"], row["file2"]
    keypoints = row["matches"]
    sorted_keypoints_matches = sorted(keypoints, key=lambda x: x[2], reverse=True)

    kp1 = os.path.join(patches_key_dec_cache, file1) + ".pkl"
    kp2 = os.path.join(patches_key_dec_cache, file2) + ".pkl"

    # Load keypoints from pkl files
    keypoints1 = load_keypoints(kp1)
    keypoints2 = load_keypoints(kp2)

    # Get patch information
    patch1_info = get_patch_info(
        base_path,
        os.path.basename(file1).split("_")[0],
        os.path.basename(file1).split("_")[1].split(".")[0],
    )
    patch2_info = get_patch_info(
        base_path,
        os.path.basename(file2).split("_")[0],
        os.path.basename(file2).split("_")[1].split(".")[0],
    )

    if patch1_info is None or patch2_info is None:
        print("Couldn't load patch info for one of the matches. Skipping...")
        return

    file1_pathch_path = os.path.join(
        base_path, os.path.basename(file1).split("_")[0], file1
    )
    file2_pathch_path = os.path.join(
        base_path, os.path.basename(file2).split("_")[0], file2
    )

    # Load patch images
    patch1 = load_image(file1_pathch_path)
    patch2 = load_image(file2_pathch_path)

    if patch1 is None or patch2 is None:
        return

    # Load original images
    img1_name = os.path.basename(file1).split("_")[0] + ".jpg"
    img2_name = os.path.basename(file2).split("_")[0] + ".jpg"

    img1_path = os.path.join(image_path, img1_name)
    img2_path = os.path.join(image_path, img2_name)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    if img1 is None or img2 is None:
        return

    # Create a new figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

    # Display the original images
    ax1.imshow(img1)
    ax2.imshow(img2)

    # Display the patch images
    ax3.imshow(patch1)
    ax4.imshow(patch2)

    # Draw rectangles around patches on original images
    rect1 = plt.Rectangle(
        (patch1_info["coordinates"][0], patch1_info["coordinates"][1]),
        patch1_info["coordinates"][2] - patch1_info["coordinates"][0],
        patch1_info["coordinates"][3] - patch1_info["coordinates"][1],
        fill=False,
        edgecolor="yellow",
        linewidth=2,
    )
    rect2 = plt.Rectangle(
        (patch2_info["coordinates"][0], patch2_info["coordinates"][1]),
        patch2_info["coordinates"][2] - patch2_info["coordinates"][0],
        patch2_info["coordinates"][3] - patch2_info["coordinates"][1],
        fill=False,
        edgecolor="yellow",
        linewidth=2,
    )
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_keypoints_matches)))

    # Plot keypoints on the patches
    for ii, kep_match in enumerate(sorted_keypoints_matches):
        kp1 = keypoints1[kep_match[0]][0]
        kp2 = keypoints2[kep_match[1]][0]

        ax3.scatter(kp1[0], kp1[1], c="blue", marker="x")
        ax4.scatter(kp2[0], kp2[1], c="blue", marker="x")

        # Convert data coordinates to figure coordinates
        # Get points in display coordinates
        p1 = ax3.transData.transform((kp1[0], kp1[1]))
        p2 = ax4.transData.transform((kp2[0], kp2[1]))

        # Convert to figure coordinates
        p1_fig = fig.transFigure.inverted().transform(p1)
        p2_fig = fig.transFigure.inverted().transform(p2)

        # Draw line in figure coordinates
        line = Line2D(
            [p1_fig[0], p2_fig[0]],
            [p1_fig[1], p2_fig[1]],
            transform=fig.transFigure,
            alpha=0.5,
        )
        fig.lines.append(line)
        if ii > 200:
            break

    # Set titles
    fig.suptitle("Match Visualization", fontsize=20)
    ax1.set_title(f"Original Image 1: {img1_name}", fontsize=14)
    ax2.set_title(f"Original Image 2: {img2_name}", fontsize=14)
    ax3.set_title(f"Patch 1: {os.path.basename(file1)}", fontsize=14)
    ax4.set_title(f"Patch 2: {os.path.basename(file2)}", fontsize=14)

    # Remove axes for better visualization
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis("off")

    # plt.tight_layout()
    if debug:
        plt.show()
    else:
        st.pyplot(fig)


# Main function for visualization
def main():
    from dotenv import load_dotenv

    load_dotenv()
    DEBUG = os.getenv("DEBUG", "False").lower() in ["true", "1", "t"]

    base_path = os.getenv("BASE_PATH")
    IMAGES_IN_path = os.path.join(base_path, os.getenv("IMAGES_IN"))
    PATCHES_IN = os.path.join(base_path, os.getenv("PATCHES_IN"))

    # A csv file with matches (patches matched)
    _sift_matches_w_tp = os.getenv("SIFT_MATCHES_W_TP")
    csv_sift_matches_w_tp_w_homo = os.path.join(
        base_path, os.getenv("SIFT_MATCHES_W_TP_W_HOMO")
    )

    patches_key_dec_cache = os.path.join(base_path, os.getenv("PATCHES_CACHE"))

    # Define output filename * file with raw for each match
    input_main_csv_file = os.path.join(base_path, csv_sift_matches_w_tp_w_homo)

    if input_main_csv_file is not None:
        # Debug flag to visualize the first match automatically
        if DEBUG == 1:
            print("Debug mode is ON: Displaying the first match in the file.")
            df = pd.read_csv(input_main_csv_file)
            df["matches"] = df["matches"].apply(ast.literal_eval)
            visualize_match(
                df.iloc[10], PATCHES_IN, IMAGES_IN_path, patches_key_dec_cache, DEBUG
            )
        else:
            # Streamlit UI
            st.title("Image Matches Visualization")
            # Read CSV and parse it
            df = pd.read_csv(input_main_csv_file)

            # Sort the DataFrame by the third column (number of matches) in descending order
            df = df.sort_values(by=df.columns[8], ascending=True)

            # Ensure that match keypoints are correctly parsed
            df["matches"] = df["matches"].apply(ast.literal_eval)
            if DEBUG == 2:
                match_index = 2
            else:
                match_index = st.number_input(
                    "Select match index to visualize",
                    min_value=0,
                    max_value=len(df) - 1,
                    step=1,
                )
            row = df.iloc[match_index]

            # Display additional information in Streamlit
            st.write("### Match Information")
            st.write(f"**sum_homo_err**: {row['sum_homo_err']}")
            st.write(f"**len_homo_err**: {row['len_homo_err']}")
            st.write(f"**mean_homo_err**: {row['mean_homo_err']}")
            st.write(f"**std_homo_err**: {row['std_homo_err']}")
            st.write(f"**Match**: {row['Match']}")

            # Visualize selected match
            visualize_match(
                row, PATCHES_IN, IMAGES_IN_path, patches_key_dec_cache, DEBUG
            )


if __name__ == "__main__":
    main()
