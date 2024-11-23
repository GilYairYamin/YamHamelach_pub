import streamlit as st
import pandas as pd
import ast
import cv2
import os
import matplotlib.pyplot as plt
from typing import Tuple
import tempfile
import json
import pickle
import numpy as np
from matplotlib.lines import Line2D
import dropbox
from io import BytesIO
import streamlit as st
from dropbox.exceptions import ApiError, AuthError
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DropboxHandler:
    def __init__(self, access_token):
        self.access_token = access_token
        self._dbx = None

    @property
    def dbx(self):
        if self._dbx is None:
            try:
                self._dbx = dropbox.Dropbox(self.access_token)
                # Test the connection
                self._dbx.users_get_current_account()
            except AuthError as e:
                st.error("Invalid Dropbox access token. Please check your credentials.")
                logger.error(f"Dropbox authentication error: {e}")
                raise
            except Exception as e:
                st.error("Error connecting to Dropbox.")
                logger.error(f"Dropbox connection error: {e}")
                raise
        return self._dbx

    def load_csv(self, dropbox_path):
        try:
            _, res = self.dbx.files_download(dropbox_path)
            csv_file = BytesIO(res.content)
            return pd.read_csv(csv_file)
        except ApiError as e:
            st.error(f"Error loading CSV from Dropbox: {dropbox_path}")
            logger.error(f"Dropbox API error: {e}")
            raise
        except Exception as e:
            st.error(f"Unexpected error loading CSV: {e}")
            logger.error(f"Error loading CSV: {e}")
            raise

    def load_file(self, dropbox_path):
        print(f" Loading file from Dropbox: {dropbox_path}...")
        try:
            _, res = self.dbx.files_download(dropbox_path)
            return BytesIO(res.content)
        except Exception as e:
            logger.error(f"Error loading file from Dropbox: {e}")
            raise

def load_image_from_bytes(image_bytes: BytesIO):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Failed to decode image")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None

def load_keypoints_from_bytes(pkl_bytes: BytesIO):
    try:
        data = pickle.load(pkl_bytes)
        return data['keypoints']
    except Exception as e:
        logger.error(f"Error loading keypoints: {e}")
        raise


# Function to load an image and convert to RGB for display
def load_image(image_path: str):
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(f"Could not load image: {image_path}")
        return None


def load_keypoints(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    return data['keypoints']

# Function to get patch information from a JSON file
def get_patch_info(base_path: str, file_name: str, box: str):
    json_file = os.path.join(base_path, file_name, f"{file_name}_patch_info.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            patch_info = json.load(f)
        return patch_info.get(box)
    return None

# Function to load a CSV file from Dropbox
def load_csv_from_dropbox(dbx, dropbox_path):
    _, res = dbx.files_download(dropbox_path)
    csv_file = BytesIO(res.content)
    df = pd.read_csv(csv_file)
    return df

# Function to visualize a match between two patches
def visualize_match(row, base_path, image_path, patches_key_dec_cache, debug=False):
    dropbox_handler = DropboxHandler(st.secrets["DROPBOX_ACCESS_TOKEN"])

    file1, file2 = row['file1'], row['file2']
    keypoints = row['matches']
    sorted_keypoints_matches = sorted(keypoints, key=lambda x: x[2], reverse=True)

    kp1 = os.path.join(patches_key_dec_cache, file1) + '.pkl'
    kp2 = os.path.join(patches_key_dec_cache, file2) + '.pkl'
    keypoints1 = load_keypoints_from_bytes(dropbox_handler.load_file(kp1))
    keypoints2 = load_keypoints_from_bytes(dropbox_handler.load_file(kp2))

    # # Load keypoints from pkl files
    # keypoints1 = load_keypoints(kp1)
    # keypoints2 = load_keypoints(kp2)

    # Get patch information
    patch1_info = get_patch_info(base_path, os.path.basename(file1).split('_')[0],
                                 os.path.basename(file1).split('_')[1].split('.')[0])
    patch2_info = get_patch_info(base_path, os.path.basename(file2).split('_')[0],
                                 os.path.basename(file2).split('_')[1].split('.')[0])

    if patch1_info is None or patch2_info is None:
        print(f"Couldn't load patch info for one of the matches. Skipping...")
        return

    file1_pathch_path = os.path.join(base_path, os.path.basename(file1).split('_')[0], file1)
    file2_pathch_path = os.path.join(base_path, os.path.basename(file2).split('_')[0], file2)

    # Load patch images
    # patch1 = load_image(file1_pathch_path)
    # patch2 = load_image(file2_pathch_path)
    patch1 = load_image_from_bytes(dropbox_handler.load_file(file1_pathch_path))
    patch2 = load_image_from_bytes(dropbox_handler.load_file(file2_pathch_path))

    if patch1 is None or patch2 is None:
        return

    # Load original images
    img1_name = os.path.basename(file1).split('_')[0] + '.jpg'
    img2_name = os.path.basename(file2).split('_')[0] + '.jpg'

    img1_path = os.path.join(image_path, img1_name)
    img2_path = os.path.join(image_path, img2_name)

    # img1 = load_image(img1_path)
    # img2 = load_image(img2_path)
    img1 = load_image_from_bytes(dropbox_handler.load_file(img1_path))
    img2 = load_image_from_bytes(dropbox_handler.load_file(img2_path))

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
    rect1 = plt.Rectangle((patch1_info['coordinates'][0], patch1_info['coordinates'][1]),
                          patch1_info['coordinates'][2] - patch1_info['coordinates'][0],
                          patch1_info['coordinates'][3] - patch1_info['coordinates'][1],
                          fill=False, edgecolor='yellow', linewidth=2)
    rect2 = plt.Rectangle((patch2_info['coordinates'][0], patch2_info['coordinates'][1]),
                          patch2_info['coordinates'][2] - patch2_info['coordinates'][0],
                          patch2_info['coordinates'][3] - patch2_info['coordinates'][1],
                          fill=False, edgecolor='yellow', linewidth=2)
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_keypoints_matches)))

    # Plot keypoints on the patches
    for ii, kep_match in enumerate(sorted_keypoints_matches):
        kp1 = keypoints1[kep_match[0]][0]
        kp2 = keypoints2[kep_match[1]][0]

        ax3.scatter(kp1[0], kp1[1],  c='blue', marker='x')
        ax4.scatter(kp2[0], kp2[1], c='blue', marker='x')

        # Convert data coordinates to figure coordinates
        # Get points in display coordinates
        p1 = ax3.transData.transform((kp1[0], kp1[1]))
        p2 = ax4.transData.transform((kp2[0], kp2[1]))

        # Convert to figure coordinates
        p1_fig = fig.transFigure.inverted().transform(p1)
        p2_fig = fig.transFigure.inverted().transform(p2)

        # Draw line in figure coordinates
        line = Line2D([p1_fig[0], p2_fig[0]],
                                       [p1_fig[1], p2_fig[1]],
                                       transform=fig.transFigure,
                                       alpha=0.5)
        fig.lines.append(line)
        if (ii > 200):
            break

    # Set titles
    fig.suptitle(f"Match Visualization", fontsize=20)
    ax1.set_title(f"Original Image 1: {img1_name}", fontsize=14)
    ax2.set_title(f"Original Image 2: {img2_name}", fontsize=14)
    ax3.set_title(f"Patch 1: {os.path.basename(file1)}", fontsize=14)
    ax4.set_title(f"Patch 2: {os.path.basename(file2)}", fontsize=14)

    # Remove axes for better visualization
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis('off')

    #plt.tight_layout()
    if (debug):
        plt.show()
    else:
        st.pyplot(fig)

    # Display additional information in Streamlit
    st.write("### Additional Information")
    st.write(f"**sum_homo_err**: {row['sum_homo_err']}")
    st.write(f"**len_homo_err**: {row['len_homo_err']}")
    st.write(f"**mean_homo_err**: {row['mean_homo_err']}")
    st.write(f"**std_homo_err**: {row['std_homo_err']}")
    st.write(f"**is_valid**: {row['is_valid']}")
    st.write(f"**Match**: {row['Match']}")

# Main function for visualization
def main():
    from dotenv import load_dotenv

    load_dotenv()
    DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1', 't']

    # Dropbox API setup
    #DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')
    DROPBOX_ACCESS_TOKEN = st.secrets["DROPBOX_ACCESS_TOKEN"]
    base_path = st.secrets["BASE_PATH"]

    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

    input_main_csv_file = "/output.csv"


    df = load_csv_from_dropbox(dbx, input_main_csv_file)

    # Sort the DataFrame by the third column (number of matches) in descending order
    df = df.sort_values(by=df.columns[8], ascending=True)

    # Ensure that match keypoints are correctly parsed
    df['matches'] = df['matches'].apply(ast.literal_eval)

    if DEBUG == 2:
        match_index = 2
    else:
        st.title("Image Matches Visualization")
        match_index = st.number_input("Select match index to visualize", min_value=0, max_value=len(df) - 1, step=1)
    row = df.iloc[match_index]

    # Display additional information in Streamlit
    st.write("### Match Information")
    st.write(f"**sum_homo_err**: {row['sum_homo_err']}")
    st.write(f"**len_homo_err**: {row['len_homo_err']}")
    st.write(f"**mean_homo_err**: {row['mean_homo_err']}")
    st.write(f"**std_homo_err**: {row['std_homo_err']}")
    st.write(f"**is_valid**: {row['is_valid']}")
    st.write(f"**Match**: {row['Match']}")

    # Visualize selected match
    IMAGES_IN_path = os.path.join(base_path, os.getenv('IMAGES_IN'))
    PATCHES_IN = os.path.join(base_path, os.getenv('PATCHES_IN'))
    patches_key_dec_cache = os.path.join(base_path, os.getenv('PATCHES_CACHE'))
    visualize_match(row, PATCHES_IN, IMAGES_IN_path, patches_key_dec_cache, DEBUG)

if __name__ == "__main__":
    main()
