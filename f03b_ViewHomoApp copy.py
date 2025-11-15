import ast
import json
import os
import pickle
from pathlib import Path
# import tempfile
# from typing import TupleP

import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot

# import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from dotenv import load_dotenv
import plotly.graph_objects as go


# Function to load an image and convert to RGB for display
def load_image(image_path: str):
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        st.warning(f"Could not load image: {image_path}")
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


def plot_image_with_rects(img, patch_infos, title, img_name):
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    for patch_id, info in patch_infos.items():
        x0, y0, x1, y1 = info["coordinates"]
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="yellow", width=3),
            fillcolor="rgba(255,255,0,0.1)",
            name=patch_id,
        )
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0],
            text=[patch_id],
            mode="text",
            showlegend=False,
            hoverinfo="text",
            hovertext=f"Patch: {patch_id}"
        ))
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1),
        dragmode=False,
        hovermode="closest",
        height=600,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    return fig


def plot_patch_match(patch1, patch2, keypoints1, keypoints2, matches):
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
    ax3.imshow(patch1)
    ax4.imshow(patch2)
    for kep_match in matches:
        kp1 = keypoints1[kep_match[0]][0]
        kp2 = keypoints2[kep_match[1]][0]
        ax3.scatter(kp1[0], kp1[1], c="blue", marker="x")
        ax4.scatter(kp2[0], kp2[1], c="blue", marker="x")
        # Draw line between the two axes (not trivial in matplotlib, so skip for now)
    ax3.set_title("Patch 1")
    ax4.set_title("Patch 2")
    for ax in [ax3, ax4]:
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
        script_dir / '.env',
        Path.cwd() / '.env',
        Path.home() / '.env'
    ]
    
    print("\nChecking for .env files in:")
    for loc in possible_env_locations:
        print(f"- {loc}: {'EXISTS' if loc.exists() else 'NOT FOUND'}")
    
    # Load .env file from the script's directory
    env_path = script_dir / '.env'
    print("\nLoading .env from:", env_path)
    
    # Print the contents of the .env file if it exists
    if env_path.exists():
        print("\nContents of .env file:")
        with open(env_path, 'r') as f:
            print(f.read())
    else:
        print("\nWARNING: .env file not found at:", env_path)
    
    # Load environment variables
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Debug print all relevant environment variables
    print("\nEnvironment variables after loading .env:")
    for key in ["BASE_PATH", "IMAGES_IN", "PATCHES_IN", "SIFT_MATCHES_1000", 
                "SIFT_MATCHES_W_TP", "SIFT_MATCHES_W_TP_W_HOMO", "PATCHES_CACHE"]:
        print(f"{key}: {os.getenv(key)}")
    
    DEBUG = os.getenv("DEBUG", "False").lower() in ["true", "1", "t"]
    DEBUG_DISPLAY = os.getenv("DEBUG_DISPLAY", "False").lower() in ["true", "1", "t"]
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
    if (DEBUG == 1):
        input_main_csv_file = os.path.join(base_path, sift_debug_file)
    else:
        input_main_csv_file = os.path.join(base_path, csv_sift_matches_w_tp_w_homo)

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
            st.title("Interactive Patch Match Visualization")
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

            # Load original images
            file1, file2 = row["file1"], row["file2"]
            img1_name = os.path.basename(file1).split("_")[0] + ".jpg"
            img2_name = os.path.basename(file2).split("_")[0] + ".jpg"
            img1 = load_image(os.path.join(IMAGES_IN_path, img1_name))
            img2 = load_image(os.path.join(IMAGES_IN_path, img2_name))

            # Load all patch info for these images
            patch_info1_path = os.path.join(base_path, os.path.basename(file1).split("_")[0], f"{os.path.basename(file1).split('_')[0]}_patch_info.json")
            patch_info2_path = os.path.join(base_path, os.path.basename(file2).split("_")[0], f"{os.path.basename(file2).split('_')[0]}_patch_info.json")
            with open(patch_info1_path, "r") as f:
                patch_infos1 = json.load(f)
            with open(patch_info2_path, "r") as f:
                patch_infos2 = json.load(f)

            st.write("### Hover over a rectangle to see patch matches")

            col1, col2 = st.columns(2)
            with col1:
                fig1 = plot_image_with_rects(img1, patch_infos1, f"Original Image 1: {img1_name}", img1_name)
                hover_patch1 = st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = plot_image_with_rects(img2, patch_infos2, f"Original Image 2: {img2_name}", img2_name)
                hover_patch2 = st.plotly_chart(fig2, use_container_width=True)

            # --- Hover logic ---
            # Streamlit does not natively support hover callbacks, but Plotly does.
            # We can use st.session_state to store which patch is hovered, and provide a selectbox for demo.

            st.write("#### Select a patch to view keypoint matches")
            patch1_id = st.selectbox("Patch from Image 1", list(patch_infos1.keys()))
            patch2_id = st.selectbox("Patch from Image 2", list(patch_infos2.keys()))

            # Only show patch match if both are selected
            if patch1_id and patch2_id:
                # Find the row in df that matches these patches
                match_row = df[(df["file1"].str.contains(patch1_id)) & (df["file2"].str.contains(patch2_id))]
                if not match_row.empty:
                    match_row = match_row.iloc[0]
                    kp1_path = os.path.join(patches_key_dec_cache, match_row["file1"]) + ".pkl"
                    kp2_path = os.path.join(patches_key_dec_cache, match_row["file2"]) + ".pkl"
                    keypoints1 = load_keypoints(kp1_path)
                    keypoints2 = load_keypoints(kp2_path)
                    patch1_img = load_image(os.path.join(base_path, os.path.basename(match_row["file1"]).split("_")[0], match_row["file1"]))
                    patch2_img = load_image(os.path.join(base_path, os.path.basename(match_row["file2"]).split("_")[0], match_row["file2"]))
                    st.write(f"#### Keypoint matches for {patch1_id} and {patch2_id}")
                    plot_patch_match(patch1_img, patch2_img, keypoints1, keypoints2, match_row["matches"])
                else:
                    st.info("No match found for selected patches.")


if __name__ == "__main__":
    main()
