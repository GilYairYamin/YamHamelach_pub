import ast
import json
import os
import pickle
from pathlib import Path

# import tempfile
# from typing import TupleP
import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Set the backend to Agg before importing pyplot
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
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
    json_file = os.path.join(
        base_path, file_name, f"{file_name}_patch_info.json"
    )
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            patch_info = json.load(f)
        return patch_info.get(box)
    return None


# Function to visualize a match between two patches
def visualize_match(
    row, base_path, image_path, patches_key_dec_cache, debug=False
):
    file1, file2 = row["file1"], row["file2"]
    keypoints = row["matches"]
    sorted_keypoints_matches = sorted(
        keypoints, key=lambda x: x[2], reverse=True
    )

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
        plt.close("all")  # Close any existing figures
        st.pyplot(fig)
        plt.close(fig)  # Close the figure after displaying


def plot_image_with_rects(img, rects, title):
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    for rect in rects:
        x0, y0, x1, y1 = rect["coords"]
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="yellow", width=3),
            fillcolor="rgba(0,0,0,0)",
            name=rect["label"],
            opacity=0.5,
        )
        fig.add_trace(
            go.Scatter(
                x=[x0],
                y=[y0],
                text=[rect["label"]],
                mode="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=30, b=0),
        dragmode=False,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_patch_keypoints(patch1, patch2, keypoints1, keypoints2, matches):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(patch1)
    ax2.imshow(patch2)
    for idx, (i1, i2, _) in enumerate(matches):
        kp1 = keypoints1[i1][0]
        kp2 = keypoints2[i2][0]
        ax1.scatter(kp1[0], kp1[1], c="blue", marker="x")
        ax2.scatter(kp2[0], kp2[1], c="blue", marker="x")
        if idx > 200:
            break
    ax1.set_title("Patch 1")
    ax2.set_title("Patch 2")
    for ax in [ax1, ax2]:
        ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# Main function for visualization
def main():
    # Get the absolute path to the script's directory
    script_dir = Path(__file__).parent.absolute()

    # Debug information about paths
    print("Current working directory:", os.getcwd())
    print("Script directory:", script_dir)

    # Check for .env files in multiple locations
    possible_env_locations = [
        script_dir / ".env",
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]

    print("\nChecking for .env files in:")
    for loc in possible_env_locations:
        print(f"- {loc}: {'EXISTS' if loc.exists() else 'NOT FOUND'}")

    # Load .env file from the script's directory
    env_path = script_dir / ".env"
    print("\nLoading .env from:", env_path)

    # Print the contents of the .env file if it exists
    if env_path.exists():
        print("\nContents of .env file:")
        with open(env_path, "r") as f:
            print(f.read())
    else:
        print("\nWARNING: .env file not found at:", env_path)

    # Load environment variables
    load_dotenv(dotenv_path=env_path, override=True)

    # Debug print all relevant environment variables
    print("\nEnvironment variables after loading .env:")
    for key in [
        "BASE_PATH",
        "IMAGES_IN",
        "PATCHES_IN",
        "SIFT_MATCHES_1000",
        "SIFT_MATCHES_W_TP",
        "SIFT_MATCHES_W_TP_W_HOMO",
        "PATCHES_CACHE",
    ]:
        print(f"{key}: {os.getenv(key)}")

    DEBUG = os.getenv("DEBUG", "False").lower() in ["true", "1", "t"]
    DEBUG_DISPLAY = os.getenv("DEBUG_DISPLAY", "False").lower() in [
        "true",
        "1",
        "t",
    ]
    print("\nDEBUG: " + str(DEBUG))
    print("DEBUG_DISPLAY: " + str(DEBUG_DISPLAY))

    base_path = os.getenv("BASE_PATH")
    if not base_path:
        raise ValueError("BASE_PATH environment variable is not set!")
    print("\nbase_path: " + base_path)

    IMAGES_IN_path = os.path.join(base_path, os.getenv("IMAGES_IN"))
    PATCHES_IN = os.path.join(base_path, os.getenv("PATCHES_IN"))

    sift_debug_file = os.path.join(base_path, os.getenv("SIFT_MATCHES_1000"))
    # A csv file with matches (patches matched)
    _sift_matches_w_tp = os.getenv("SIFT_MATCHES_W_TP")
    csv_sift_matches_w_tp_w_homo = os.path.join(
        base_path, os.getenv("SIFT_MATCHES_W_TP_W_HOMO")
    )

    patches_key_dec_cache = os.path.join(base_path, os.getenv("PATCHES_CACHE"))

    # Define output filename * file with raw for each match
    if DEBUG == 1:
        input_main_csv_file = os.path.join(base_path, sift_debug_file)
    else:
        input_main_csv_file = os.path.join(
            base_path, csv_sift_matches_w_tp_w_homo
        )

    if input_main_csv_file is not None:
        # Debug flag to visualize the first match automatically
        if DEBUG_DISPLAY == 1:
            print("Debug mode is ON: Displaying the first match in the file.")
            df = pd.read_csv(input_main_csv_file)
            df["matches"] = df["matches"].apply(ast.literal_eval)
            visualize_match(
                df.iloc[10],
                PATCHES_IN,
                IMAGES_IN_path,
                patches_key_dec_cache,
                DEBUG_DISPLAY,
            )
        else:
            # Streamlit UI
            st.title("Interactive Patch Viewer")
            # Read CSV and parse it
            print("Reading CSV file: " + input_main_csv_file)
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

            # Load images and patch info
            file1, file2 = row["file1"], row["file2"]
            img1_name = os.path.basename(file1).split("_")[0] + ".jpg"
            img2_name = os.path.basename(file2).split("_")[0] + ".jpg"
            img1_path = os.path.join(IMAGES_IN_path, img1_name)
            img2_path = os.path.join(IMAGES_IN_path, img2_name)
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)

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

            rects1 = []
            rects2 = []
            if patch1_info:
                rects1.append(
                    {
                        "coords": patch1_info["coordinates"],
                        "label": os.path.basename(file1),
                    }
                )
            if patch2_info:
                rects2.append(
                    {
                        "coords": patch2_info["coordinates"],
                        "label": os.path.basename(file2),
                    }
                )

            # Display images with rectangles
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_image_with_rects(
                        img1, rects1, f"Original Image 1: {img1_name}"
                    ),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    plot_image_with_rects(
                        img2, rects2, f"Original Image 2: {img2_name}"
                    ),
                    use_container_width=True,
                )

            # Hover logic
            st.info(
                "Hover over a rectangle in the image to see patch keypoints. (In Streamlit, click the rectangle label below to display the patch keypoints.)"
            )

            # Simulate hover by letting user select which patch to view keypoints for
            patch_to_show = st.radio(
                "Show patch keypoints for:", [file1, file2]
            )

            if patch_to_show == file1 and patch1_info and patch2_info:
                kp1_path = (
                    os.path.join(base_path, patches_key_dec_cache, file1)
                    + ".pkl"
                )
                kp2_path = (
                    os.path.join(base_path, patches_key_dec_cache, file2)
                    + ".pkl"
                )
                keypoints1 = load_keypoints(kp1_path)
                keypoints2 = load_keypoints(kp2_path)
                patch1_path = os.path.join(
                    base_path, os.path.basename(file1).split("_")[0], file1
                )
                patch2_path = os.path.join(
                    base_path, os.path.basename(file2).split("_")[0], file2
                )
                patch1 = load_image(patch1_path)
                patch2 = load_image(patch2_path)
                st.subheader("Patch Keypoints Visualization")
                plot_patch_keypoints(
                    patch1, patch2, keypoints1, keypoints2, row["matches"]
                )
            elif patch_to_show == file2 and patch1_info and patch2_info:
                kp1_path = (
                    os.path.join(base_path, patches_key_dec_cache, file1)
                    + ".pkl"
                )
                kp2_path = (
                    os.path.join(base_path, patches_key_dec_cache, file2)
                    + ".pkl"
                )
                keypoints1 = load_keypoints(kp1_path)
                keypoints2 = load_keypoints(kp2_path)
                patch1_path = os.path.join(
                    base_path, os.path.basename(file1).split("_")[0], file1
                )
                patch2_path = os.path.join(
                    base_path, os.path.basename(file2).split("_")[0], file2
                )
                patch1 = load_image(patch1_path)
                patch2 = load_image(patch2_path)
                st.subheader("Patch Keypoints Visualization")
                plot_patch_keypoints(
                    patch1, patch2, keypoints1, keypoints2, row["matches"]
                )
            else:
                st.warning("Patch info not found for one or both patches.")


if __name__ == "__main__":
    main()
