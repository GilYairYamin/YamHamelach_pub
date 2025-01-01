import ast
import os

import pandas as pd
from dotenv import load_dotenv

# Import the visualization function from your main script
# Assuming it's saved as visualization.py
from readme_supplementary_images.BK.f04_generate_all_matches_v4 import (
    visualize_image_matches,
)


def test_problematic_pairs():
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_path = os.getenv("BASE_PATH")
    image_path = os.path.join(base_path, os.getenv("IMAGES_IN"))
    patches_path = os.path.join(base_path, os.getenv("PATCHES_IN"))
    patches_cache = os.path.join(base_path, os.getenv("PATCHES_CACHE"))
    csv_file = os.path.join(base_path, os.getenv("SIFT_MATCHES_W_TP_W_HOMO"))

    # Create output directory
    output_dir = "debug_output_v5"
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)
    df["matches"] = df["matches"].apply(ast.literal_eval)

    # Test pairs from your examples
    test_pairs = [
        ("M42163-1-E.jpg", "M42748-1-E.jpg"),
        ("M42815-1-E.jpg", "M42937-1-E.jpg"),
        ("M42968-1-E.jpg", "M43500-1-E.jpg"),
        ("M42812-1-E.jpg", "M43362-1-E.jpg"),
    ]

    # Process each pair
    for img1, img2 in test_pairs:
        print(f"\nProcessing pair: {img1} - {img2}")
        visualize_image_matches(
            img1_name=img1,
            img2_name=img2,
            matches_df=df,
            base_path=base_path,
            patches_path=patches_path,
            image_path=image_path,
            output_dir=output_dir,
            distance_threshold=100,
            debug=True,
        )


if __name__ == "__main__":
    test_problematic_pairs()
