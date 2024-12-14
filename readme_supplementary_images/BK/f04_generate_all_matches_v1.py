import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import pandas as pd
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple


def load_image(image_path: str):
    """Load an image and convert to RGB."""
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(f"Could not load image: {image_path}")
        return None


def load_keypoints(pkl_file_path: str) -> dict:
    """Load keypoints from pickle file."""
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    return data['keypoints']


def get_patch_info(base_path: str, file_name: str, box: str) -> dict:
    """Get patch information from JSON file."""
    json_file = os.path.join(base_path, file_name, f"{file_name}_patch_info.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            patch_info = json.load(f)
        return patch_info.get(box)
    return None


def visualize_all_matches(df_matches: pd.DataFrame, base_path: str, image_path: str,
                          patches_key_dec_cache: str, img1_name: str, img2_name: str):
    """
    Create a visualization showing all matching patches between two images.

    Args:
        df_matches: DataFrame containing match information
        base_path: Base path for patch files
        image_path: Path to original images
        patches_key_dec_cache: Path to keypoint cache
        img1_name: Name of first image
        img2_name: Name of second image
    """
    # Filter matches with distance > 100
    df_matches = df_matches[df_matches['distance'] > 100]

    if len(df_matches) == 0:
        print(f"No matches with distance > 100 found between {img1_name} and {img2_name}")
        return None

    # Load original images
    img1_path = os.path.join(image_path, img1_name + '.jpg')
    img2_path = os.path.join(image_path, img2_name + '.jpg')

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    if img1 is None or img2 is None:
        print("Could not load original images")
        return None

    # Create figure with two main subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display original images
    ax1.imshow(img1)
    ax2.imshow(img2)

    # Generate colors for different patches
    n_matches = len(df_matches)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_matches))

    match_cnt = 0
    # Process each match
    for idx, row in df_matches.iterrows():
        file1, file2 = row['file1'], row['file2']
        distance = row['distance']  # Get the distance value

        # Get patch info
        patch1_name = os.path.basename(file1).split('_')[0]
        patch2_name = os.path.basename(file2).split('_')[0]
        box1 = os.path.basename(file1).split('_')[1].split('.')[0]
        box2 = os.path.basename(file2).split('_')[1].split('.')[0]

        patch1_info = get_patch_info(base_path, patch1_name, box1)
        patch2_info = get_patch_info(base_path, patch2_name, box2)

        if patch1_info is None or patch2_info is None:
            continue

        # Use color from colormap for this match
        color = colors[match_cnt]

        # Draw rectangles for patches
        rect1 = plt.Rectangle((patch1_info['coordinates'][0], patch1_info['coordinates'][1]),
                              patch1_info['coordinates'][2] - patch1_info['coordinates'][0],
                              patch1_info['coordinates'][3] - patch1_info['coordinates'][1],
                              fill=False, edgecolor=color, linewidth=2)
        rect2 = plt.Rectangle((patch2_info['coordinates'][0], patch2_info['coordinates'][1]),
                              patch2_info['coordinates'][2] - patch2_info['coordinates'][0],
                              patch2_info['coordinates'][3] - patch2_info['coordinates'][1],
                              fill=False, edgecolor=color, linewidth=2)

        ax1.add_patch(rect1)
        ax2.add_patch(rect2)

        # Calculate box centers
        center1_x = (patch1_info['coordinates'][0] + patch1_info['coordinates'][2]) / 2
        center1_y = (patch1_info['coordinates'][1] + patch1_info['coordinates'][3]) / 2
        center2_x = (patch2_info['coordinates'][0] + patch2_info['coordinates'][2]) / 2
        center2_y = (patch2_info['coordinates'][1] + patch2_info['coordinates'][3]) / 2

        # Draw lines between box centers
        line1 = Line2D([patch1_info['coordinates'][0], patch1_info['coordinates'][2]],
                       [patch1_info['coordinates'][1], patch1_info['coordinates'][3]],
                       color=color, linestyle='--', alpha=0.5)
        line2 = Line2D([patch2_info['coordinates'][0], patch2_info['coordinates'][2]],
                       [patch2_info['coordinates'][1], patch2_info['coordinates'][3]],
                       color=color, linestyle='--', alpha=0.5)

        ax1.add_line(line1)
        ax2.add_line(line2)

        # Plot center points
        ax1.scatter(center1_x, center1_y, c=[color], marker='o', s=50)
        ax2.scatter(center2_x, center2_y, c=[color], marker='o', s=50)

        # Add match information as text
        ax1.annotate(f'Match {match_cnt + 1}\nDist: {distance:.1f}',
                     (center1_x, center1_y),
                     xytext=(5, 5), textcoords='offset points',
                     color=color)
        ax2.annotate(f'Match {match_cnt + 1}\nDist: {distance:.1f}',
                     (center2_x, center2_y),
                     xytext=(5, 5), textcoords='offset points',
                     color=color)
        match_cnt += 1

    # Set titles and remove axes
    ax1.set_title(f"Image 1: {img1_name}", fontsize=14)
    ax2.set_title(f"Image 2: {img2_name}", fontsize=14)
    ax1.axis('off')
    ax2.axis('off')

    # Add overall title with match count
    plt.suptitle(f"Matching Patches Between Images (Distance > 100, {match_cnt} matches)", fontsize=16)
    plt.tight_layout()

    return fig


def main():
    # Example usage
    from dotenv import load_dotenv
    import pandas as pd

    load_dotenv()

    base_path = os.getenv('BASE_PATH')
    IMAGES_IN_path = os.path.join(base_path, os.getenv('IMAGES_IN'))
    PATCHES_IN = os.path.join(base_path, os.getenv('PATCHES_IN'))
    patches_key_dec_cache = os.path.join(base_path, os.getenv('PATCHES_CACHE'))
    csv_file = os.path.join(base_path, os.getenv('SIFT_MATCHES_W_TP_W_HOMO'))

    # Read and process CSV
    df = pd.read_csv(csv_file)

    # Get unique image pairs
    img_pairs = set()
    for _, row in df.iterrows():
        img1 = os.path.basename(row['file1']).split('_')[0]
        img2 = os.path.basename(row['file2']).split('_')[0]
        img_pairs.add((img1, img2))

    # Process each image pair
    for img1_name, img2_name in img_pairs:
        # Filter matches for this image pair
        pair_matches = df[
            (df['file1'].str.contains(img1_name)) &
            (df['file2'].str.contains(img2_name))
            ]

        if len(pair_matches) > 0:
            fig = visualize_all_matches(
                pair_matches,
                PATCHES_IN,
                IMAGES_IN_path,
                patches_key_dec_cache,
                img1_name,
                img2_name
            )

            if fig is not None:
                # Save the visualization
                output_dir = "patch_visualizations"
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, f"{img1_name}_vs_{img2_name}_patches_dist100.jpg"))
                plt.close(fig)


if __name__ == "__main__":
    main()