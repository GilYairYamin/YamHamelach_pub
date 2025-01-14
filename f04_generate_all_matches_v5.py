import argparse
import gc
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from env_arguments_loader import load_env_arguments

# from db_manager import db_manager


class MatchDebugger:
    def __init__(self, debug=False):
        self.debug = debug

    def log_image_info(self, img1_name, img2_name, img1, img2):
        if not self.debug:
            return

        print("\nImage dimensions:")
        print(f"{img1_name}: {img1.shape}")
        print(f"{img2_name}: {img2.shape}")

    def log_match_info(self, idx, file1, file2, distance):
        if not self.debug:
            return

        print(f"\nProcessing match {idx}:")
        print(f"File1: {file1}")
        print(f"File2: {file2}")
        print(f"Distance: {distance}")

    def log_patch_info(self, patch1_info, patch2_info):
        if not self.debug:
            return

        print(f"Patch1 coordinates: {patch1_info['coordinates']}")
        print(f"Patch2 coordinates: {patch2_info['coordinates']}")

    def log_centers(self, center1, center2):
        if not self.debug:
            return

        print(f"Center1 (normalized): {center1}")
        print(f"Center2 (normalized): {center2}")

    def log_figure_coords(self, fig_coords1, fig_coords2):
        if not self.debug:
            return

        coords_str1 = f"({fig_coords1[0]:.3f}, {fig_coords1[1]:.3f})"
        coords_str2 = f"({fig_coords2[0]:.3f}, {fig_coords2[1]:.3f})"
        print(f"Figure coordinates 1: {coords_str1}")
        print(f"Figure coordinates 2: {coords_str2}")

    def draw_debug_markers(self, ax1, ax2, center1, center2, w1, h1, w2, h2):
        if not self.debug:
            return

        # Draw center points as red X's
        ax1.plot(center1[0] * w1, center1[1] * h1, "rx", markersize=10)
        ax2.plot(center2[0] * w2, center2[1] * h2, "rx", markersize=10)


def load_image(image_path: str):
    """Load and convert image to RGB."""
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(f"Could not load image: {image_path}")
        return None


def get_patch_info(base_path: str, file_name: str, box: str) -> Dict:
    """Get patch coordinates from JSON file."""
    json_file = os.path.join(
        base_path, file_name, f"{file_name}_patch_info.json"
    )
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            patch_info = json.load(f)
        return patch_info.get(box)
    return None


def get_patch_center(coordinates: List[float]) -> Tuple[float, float]:
    """Calculate center point of a patch."""
    x_center = (coordinates[0] + coordinates[2]) / 2
    y_center = (coordinates[1] + coordinates[3]) / 2
    return (x_center, y_center)


def normalize_coordinates(coords, width, height, is_vertical):
    """Normalize coordinates based on image orientation"""
    if is_vertical:
        return (coords[0] / width, coords[1] / height)
    else:
        return (coords[0] / width, coords[1] / height)


def create_result_img_figure(img1, img2, img1_name, img2_name):
    # Determine image orientations

    img1_shape = img1.shape[:2]
    img2_shape = img2.shape[:2]

    is_img1_vertical = img1_shape[0] > img1_shape[1]
    is_img2_vertical = img2_shape[0] > img2_shape[1]

    # Create figure with adaptive size based on orientations
    fig = None
    if is_img1_vertical and is_img2_vertical:
        fig = plt.figure(figsize=(12, 16))
    elif not is_img1_vertical and not is_img2_vertical:
        fig = plt.figure(figsize=(20, 8))
    else:
        fig = plt.figure(figsize=(16, 12))

    img1_width_ratio = img1_shape[1] / max(img1_shape[1], img1_shape[0])
    img2_width_ratio = img2_shape[1] / max(img2_shape[1], img2_shape[0])

    gs = plt.GridSpec(
        1, 2, width_ratios=[img1_width_ratio, img2_width_ratio], figure=fig
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.set_title(f"Image 1: {img1_name}", fontsize=12)
    ax2.set_title(f"Image 2: {img2_name}", fontsize=12)

    # Display images
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    return (
        fig,
        ax1,
        ax2,
        img1_shape,
        img2_shape,
        is_img1_vertical,
        is_img2_vertical,
    )


def get_image_patches(
    matches_df: pd.DataFrame,
    img1_name: str,
    img2_name: str,
    distance_threshold: int,
    match_metric: str = "mean_homo_err",
):
    image_matches = matches_df[
        (
            (matches_df["file1"].str.startswith(img1_name.split(".")[0]))
            & (matches_df["file2"].str.startswith(img2_name.split(".")[0]))
        )
        | (
            (matches_df["file2"].str.startswith(img1_name.split(".")[0]))
            & (matches_df["file1"].str.startswith(img2_name.split(".")[0]))
        )
    ]

    # image_matches = image_matches[
    #     image_matches[match_metric] <= distance_threshold
    # ]
    match_scores = image_matches[match_metric].tolist()

    return image_matches, match_scores


def get_patches_info(row, img1_name, patches_path):
    file1, file2 = row["file1"], row["file2"]
    if row["file2"].startswith(img1_name.split(".")[0]):
        file1, file2 = file2, file1

    # Get patch information
    patch1_name = os.path.basename(file1).split("_")[0]
    patch2_name = os.path.basename(file2).split("_")[0]
    patch1_box = os.path.basename(file1).split("_")[1].split(".")[0]
    patch2_box = os.path.basename(file2).split("_")[1].split(".")[0]

    patch1_info = get_patch_info(patches_path, patch1_name, patch1_box)
    patch2_info = get_patch_info(patches_path, patch2_name, patch2_box)

    return patch1_info, patch2_info


def visualize_image_matches(
    img1_name: str,
    img2_name: str,
    matches_df: pd.DataFrame,
    base_path: str,
    patches_path: str,
    image_path: str,
    output_dir: str,
    distance_threshold: int = 25,
    homo_err_threshold: float = 100,
    match_metric: str = "mean_homo_err",
    debug: bool = False,
):
    # Initialize debugger
    debugger = MatchDebugger(debug)

    filtered_df = matches_df[
        (matches_df["distance"] >= distance_threshold)
        & (matches_df[match_metric] <= homo_err_threshold)
    ]

    # Load images
    img1 = load_image(os.path.join(image_path, img1_name))
    img2 = load_image(os.path.join(image_path, img2_name))

    if img1 is None or img2 is None:
        return

    debugger.log_image_info(img1_name, img2_name, img1, img2)

    image_matches, match_scores = get_image_patches(
        filtered_df, img1_name, img2_name, homo_err_threshold
    )

    if image_matches.empty:
        print("didn't find matches", flush=True)
        return

    # csv_name = f"{img1_name}_vs_{img2_name}.csv"
    # csv_path = os.path.join(output_dir, csv_name)
    # image_matches.to_csv(csv_path)

    (
        fig,
        ax1,
        ax2,
        img1_shape,
        img2_shape,
        is_img1_vertical,
        is_img2_vertical,
    ) = create_result_img_figure(img1, img2, img1_name, img2_name)

    # Process matches
    if match_scores:
        # Take up to first 4 scores
        scores_to_average = match_scores[: min(4, len(match_scores))]
        avg_score = round(sum(scores_to_average) / len(scores_to_average))
    else:
        avg_score = 0

    distance = -1
    for idx, row in image_matches.iterrows():
        distance = row[match_metric]
        debugger.log_match_info(idx, img1_name, img2_name, distance)

        patch1_info, patch2_info = get_patches_info(
            row, img1_name, patches_path
        )

        if patch1_info is None or patch2_info is None:
            continue

        debugger.log_patch_info(patch1_info, patch2_info)

        # Draw rectangles
        color = (
            "yellow"
            if not debug
            else np.random.rand(
                3,
            )
        )

        rect1 = Rectangle(
            (patch1_info["coordinates"][0], patch1_info["coordinates"][1]),
            patch1_info["coordinates"][2] - patch1_info["coordinates"][0],
            patch1_info["coordinates"][3] - patch1_info["coordinates"][1],
            fill=False,
            edgecolor=color,
            linewidth=1,
        )

        rect2 = Rectangle(
            (patch2_info["coordinates"][0], patch2_info["coordinates"][1]),
            patch2_info["coordinates"][2] - patch2_info["coordinates"][0],
            patch2_info["coordinates"][3] - patch2_info["coordinates"][1],
            fill=False,
            edgecolor=color,
            linewidth=1,
        )

        ax1.add_patch(rect1)
        ax2.add_patch(rect2)

        # Calculate centers in data coordinates
        center1_x = (
            patch1_info["coordinates"][0] + patch1_info["coordinates"][2]
        ) / 2
        center1_y = (
            patch1_info["coordinates"][1] + patch1_info["coordinates"][3]
        ) / 2
        center2_x = (
            patch2_info["coordinates"][0] + patch2_info["coordinates"][2]
        ) / 2
        center2_y = (
            patch2_info["coordinates"][1] + patch2_info["coordinates"][3]
        ) / 2

        # Transform to normalized device coordinates
        center1_norm = normalize_coordinates(
            (center1_x, center1_y),
            img1_shape[1],
            img1_shape[0],
            is_img1_vertical,
        )
        center2_norm = normalize_coordinates(
            (center2_x, center2_y),
            img2_shape[1],
            img2_shape[0],
            is_img2_vertical,
        )

        debugger.log_centers(center1_norm, center2_norm)

        # Get display coordinates
        bbox1 = ax1.get_position()
        bbox2 = ax2.get_position()

        # Calculate figure coordinates with orientation handling
        if is_img1_vertical:
            fig_y1 = bbox1.y0 + (1 - center1_norm[1]) * bbox1.height
        else:
            fig_y1 = bbox1.y0 + (1 - center1_norm[1]) * bbox1.height

        if is_img2_vertical:
            fig_y2 = bbox2.y0 + (1 - center2_norm[1]) * bbox2.height
        else:
            fig_y2 = bbox2.y0 + (1 - center2_norm[1]) * bbox2.height

        fig_x1 = bbox1.x0 + center1_norm[0] * bbox1.width
        fig_x2 = bbox2.x0 + center2_norm[0] * bbox2.width

        debugger.log_figure_coords((fig_x1, fig_y1), (fig_x2, fig_y2))

        # Draw line
        line = plt.Line2D(
            [fig_x1, fig_x2],
            [fig_y1, fig_y2],
            transform=fig.transFigure,
            color=color,
            alpha=0.5,
            linestyle="-",
            linewidth=1,
        )
        fig.lines.append(line)

        # Add distance text
        text_x = (fig_x1 + fig_x2) / 2
        text_y = (fig_y1 + fig_y2) / 2
        plt.figtext(
            text_x,
            text_y,
            f"{distance:.2f}",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Set titles and remove axes
    ax1.axis("off")
    ax2.axis("off")

    base_name1 = img1_name.split(".")[0]
    base_name2 = img2_name.split(".")[0]

    name_base = f"{base_name1}_{base_name2}_s{avg_score}"
    name_ending = f"patches_dist{distance:.2f}.jpg"
    file_name = f"{name_base}_{name_ending}.jpg"

    output_file = os.path.join(output_dir, file_name)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def find_unique_pairs(
    matches_df: pd.DataFrame,
    image_list: list[str] = None,
):
    image_pairs = set()
    if image_list is not None:
        image_pairs = set(
            [
                tuple(sorted([img1, img2]))
                for img1 in image_list
                for img2 in image_list
                if img1 != img2
            ]
        )
        return image_pairs

    def remove_extenstion(val):
        return os.path.basename(val).split("_")[0] + ".jpg"

    unique_pairs = matches_df[["file1", "file2"]]
    unique_pairs.loc[:, "file1"] = unique_pairs["file1"].apply(
        remove_extenstion
    )
    unique_pairs.loc[:, "file2"] = unique_pairs["file2"].apply(
        remove_extenstion
    )
    unique_pairs.drop_duplicates(inplace=True)

    unique_pairs = unique_pairs[unique_pairs["file1"] != unique_pairs["file2"]]
    print(f"found {len(unique_pairs)} combinations")

    # Process all unique image pairs
    for _, row in unique_pairs.iterrows():
        pair = tuple(sorted([row["file1"], row["file2"]]))
        image_pairs.add(pair)
        # print(f"found {pair}")

    return image_pairs


def loop_over_csv(
    matches_df: pd.DataFrame,
    base_path: str,
    patches_path: str,
    image_path: str,
    output_dir: str,
    distance_threshold: int = 25,
    homo_err_threshold: float = 100,
    image_list: list[str] = None,
    match_metric: str = "mean_homo_err",
    debug: bool = False,
):
    image_pairs = find_unique_pairs(matches_df, image_list)

    print(f"Found {len(image_pairs)} unique image pairs")
    for img1, img2 in image_pairs:
        print(f"Processing pair: {img1} - {img2}")
        visualize_image_matches(
            img1,
            img2,
            matches_df,
            base_path,
            patches_path,
            image_path,
            output_dir,
            distance_threshold,
            homo_err_threshold,
            match_metric=match_metric,
            debug=debug,
        )
        gc.collect()


def load_arguments():
    args: SimpleNamespace = load_env_arguments()

    args.matches_df = pd.read_csv(args.csv_file)

    parser = argparse.ArgumentParser(description="Visualize image matches")
    parser.add_argument(
        "--output_dir",
        default=f"{args.output_base_path}/match_visualizations_5",
        help="Output directory for visualizations",
    )

    # sum_homo_err, len_homo_err, mean_homo_err, std_homo_err
    parser.add_argument(
        "--match_metric",
        default="mean_homo_err",
        help="The metric used for determining distance.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=20,
        help="Distance threshold for filtering matches",
    )
    parser.add_argument(
        "--homo_err_threshold",
        type=float,
        default=50,
        help="Distance threshold for filtering matches",
    )
    parser.add_argument(
        "--image1",
        help="Optional: specific first image to process",
        default="M42970-1-E.jpg",
    )
    parser.add_argument(
        "--image2",
        help="Optional: specific second image to process",
        default="M43003-1-E.jpg",
    )

    parser.add_argument(
        "--debug", default=False, help="Should log debug information."
    )

    parsed_args = parser.parse_args()
    return SimpleNamespace(**parsed_args.__dict__, **args.__dict__)


def main():
    # Setup argument parser for optional parameters
    args = load_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image1 and args.image2:
        # Process specific image pair
        visualize_image_matches(
            args.image1,
            args.image2,
            args.matches_df,
            args.base_path,
            args.patches_path,
            args.image_path,
            args.output_dir,
            args.distance_threshold,
            args.homo_err_threshold,
            match_metric=args.match_metric,
            debug=args.debug,
        )
        return

    try:
        loop_over_csv(
            args.matches_df,
            args.base_path,
            args.patches_path,
            args.image_path,
            args.output_dir,
            args.distance_threshold,
            args.homo_err_threshold,
            args.pam_files_to_process,
            match_metric=args.match_metric,
            debug=args.debug,
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
