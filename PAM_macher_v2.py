
from tqdm import tqdm  # Progress bar
import numpy as np

import cv2
import pickle
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json

class NaiveImageMatcher:
    def __init__(self, cache_file: str = "image_cache.pkl"):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def _get_image_key(self, file_path: str) -> str:
        return os.path.basename(file_path)

    def _serialize_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[Tuple]:
        return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    def _deserialize_keypoints(self, keypoints_data: List[Tuple]) -> List[cv2.KeyPoint]:
        return [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1], angle=pt[2],
                             response=pt[3], octave=pt[4], class_id=pt[5]) for pt in keypoints_data]

    def _process_image(self, file_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        image_key = self._get_image_key(file_path)

        if image_key in self.cache:
            print(f"Loading cached data for {file_path}")
            cached_data = self.cache[image_key]
            return (self._deserialize_keypoints(cached_data['keypoints']),
                    cached_data['descriptors'])

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        self.cache[image_key] = {
            'keypoints': self._serialize_keypoints(keypoints),
            'descriptors': descriptors
        }
        self._save_cache()

        return keypoints, descriptors

    def calc_matches(self, file1: str, file2: str) -> Tuple[
        List[cv2.KeyPoint], List[cv2.KeyPoint], np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
        kp1, des1 = self._process_image(file1)
        kp2, des2 = self._process_image(file2)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx, m.distance))

        return kp1, kp2, des1, des2, good_matches

    def is_match(self, file1: str, file2: str, threshold: int = 10) -> bool:
        _, _, _, _, good_matches = self.calc_matches(file1, file2)
        return len(good_matches) >= threshold


if __name__ == "__main1__":
    # Example usage:
    matcher = NaiveImageMatcher()
    result = matcher.is_match('image1.jpg', 'image2.jpg')
    if result:
        print("The images match!")
    else:
        print("The images do not match.")

import os
import pandas as pd
import csv

import os
import pandas as pd
import csv


class FragmentMatcher:
    # def __init__(self, pam_csv_file, image_base_path):
    #     self.pam_csv_file = pam_csv_file
    #     self.image_base_path = image_base_path
    #     self.matcher = NaiveImageMatcher()
    #     self.missing_files = []  # To store missing files
    def __init__(self, pam_csv_file, image_base_path, original_image_path):
        self.pam_csv_file = pam_csv_file
        self.image_base_path = image_base_path
        self.original_image_path = original_image_path
        self.matcher = NaiveImageMatcher()
        self.missing_files = []


    def get_patch_info(self, file_name, box):
        """Load patch information from the JSON file."""
        json_file = os.path.join(self.image_base_path, file_name, f"{file_name}_patch_info.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                patch_info = json.load(f)
            return patch_info.get(str(int(box)))
        return None

    def read_csv(self):
        """Read the CSV file and return it as a pandas DataFrame."""
        return pd.read_csv(self.pam_csv_file)

        def get_image_path(self, file_name, box):
            """Generate the image path based on file and box information."""

            image_file = f"{file_name}_{int(box)}.jpg"
            return os.path.join(self.image_base_path, file_name, image_file)

    def get_matched_pam_files(self):
        """Find matches based on Scroll and Fragment with different boxes."""
        df = self.read_csv()

        # Remove rows where Box is NaN
        df = df.dropna(subset=['Box'])

        # Perform an inner join on the Scroll and Frg columns
        merged_df = pd.merge(df, df, on=['Scroll', 'Frg'])

        # Filter the results where the Box values are different
        filtered_df = merged_df[merged_df['Box_x'] != merged_df['Box_y']]

        # Select only the necessary columns: file1, box1, file2, box2
        result_df = filtered_df[['Scroll' , 'Frg' , 'File_x', 'Box_x', 'File_y', 'Box_y']]
        return result_df

    def calculate_distances(self, matches_df):
        distances = []

        for index, row in tqdm(matches_df.iterrows(), total=matches_df.shape[0], desc="Processing Matches"):
            image_path1 = self.get_image_path(row['File_x'], row['Box_x'])
            image_path2 = self.get_image_path(row['File_y'], row['Box_y'])

            if os.path.exists(image_path1) and os.path.exists(image_path2):
                kp1, kp2, des1, des2, good_matches = self.matcher.calc_matches(image_path1, image_path2)
                distances.append({
                    'scroll': row['Scroll'],
                    'fragment': row['Frg'],
                    'file1': image_path1,
                    'file2': image_path2,
                    'distance': len(good_matches),
                    'keypoints1': [(kp.pt[0], kp.pt[1]) for kp in kp1],
                    'keypoints2': [(kp.pt[0], kp.pt[1]) for kp in kp2],
                    'matches': [(m[0], m[1]) for m in good_matches]
                })
            else:
                self.missing_files.append({
                    'scroll': row['Scroll'],
                    'fragment': row['Frg'],
                    'file1': image_path1,
                    'file2': image_path2
                })

        return distances

    def visualize_matches(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for i, match in enumerate(distances):
            # Load original images
            img1_name = os.path.basename(match['file1']).split('_')[0] + '.jpg'
            img2_name = os.path.basename(match['file2']).split('_')[0] + '.jpg'

            img1_path = os.path.join(self.original_image_path, img1_name)
            img2_path = os.path.join(self.original_image_path, img2_name)

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print(f"Couldn't load images for match {i + 1}. Skipping...")
                continue

            # Convert to RGB for matplotlib
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # Get patch information
            patch1_info = self.get_patch_info(os.path.basename(match['file1']).split('_')[0],
                                              os.path.basename(match['file1']).split('_')[1].split('.')[0])
            patch2_info = self.get_patch_info(os.path.basename(match['file2']).split('_')[0],
                                              os.path.basename(match['file2']).split('_')[1].split('.')[0])

            if patch1_info is None or patch2_info is None:
                print(f"Couldn't load patch info for match {i + 1}. Skipping...")
                continue

            # Load patch images
            patch1 = cv2.imread(match['file1'])
            patch2 = cv2.imread(match['file2'])
            patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB)
            patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)

            # Create a new figure with 4 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

            # Display the original images
            ax1.imshow(img1)
            ax2.imshow(img2)

            # Display the patch images
            ax3.imshow(patch1)
            ax4.imshow(patch2)

            # Plot matching keypoints on original images and patches
            for m in match['matches']:
                kp1 = match['keypoints1'][m[0]]
                kp2 = match['keypoints2'][m[1]]

                # Adjust keypoint coordinates for original images
                pt1_orig = (kp1[0] + patch1_info['coordinates'][0], kp1[1] + patch1_info['coordinates'][1])
                pt2_orig = (kp2[0] + patch2_info['coordinates'][0], kp2[1] + patch2_info['coordinates'][1])

                # Plot keypoints on original images
                ax1.plot(pt1_orig[0], pt1_orig[1], 'ro', markersize=3)
                ax2.plot(pt2_orig[0], pt2_orig[1], 'ro', markersize=3)

                # Plot keypoints on patches
                ax3.plot(kp1[0], kp1[1], 'ro', markersize=3)
                ax4.plot(kp2[0], kp2[1], 'ro', markersize=3)

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

            # Set titles
            fig.suptitle(f"Match Score: {match['distance']}", fontsize=20)
            ax1.set_title(f"Original Image 1: {img1_name}", fontsize=14)
            ax2.set_title(f"Original Image 2: {img2_name}", fontsize=14)
            ax3.set_title(f"Patch 1: {os.path.basename(match['file1'])}", fontsize=14)
            ax4.set_title(f"Patch 2: {os.path.basename(match['file2'])}", fontsize=14)

            # Remove axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axis('off')

            # Adjust layout and save the figure
            plt.tight_layout()
            output_file = os.path.join(output_dir, f"match_{i + 1}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def write_csv(self, distances, success_csv, missing_csv):
        """Write the results to two CSV files."""
        # Write the successful match results to CSV
        with open(success_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['scroll', 'fragment', 'file1', 'file2', 'distance'])
            writer.writeheader()
            for match in distances:
                writer.writerow({
                    'scroll': match['scroll'],
                    'fragment': match['fragment'],
                    'file1': match['file1'],
                    'file2': match['file2'],
                    'distance': match['distance']
                })

        # Write the missing file information to another CSV
        with open(missing_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['scroll', 'fragment', 'file1', 'file2'])
            writer.writeheader()
            writer.writerows(self.missing_files)
    def run(self, success_csv='matches.csv', missing_csv='missing_files.csv', output_dir='match_visualizations'):
        matches_df = self.get_matched_pam_files()
        distances = self.calculate_distances(matches_df)
        self.write_csv(distances, success_csv, missing_csv)
        self.visualize_matches(distances, output_dir)
        print(f"Results written to {success_csv} and missing file info to {missing_csv}")
        print(f"Match visualizations saved in {output_dir}")

# Update the main script to include the path to original images
if __name__ == "__main__":
    csv_file = "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/PAM_Identification_for_Asaf_Round_1_avinoamEdit.csv"
    image_base_path = "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/bounding_boxes_crops/patches/"
    original_image_path = "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/input_dead_see_images/"
    matcher = FragmentMatcher(csv_file, image_base_path, original_image_path)
    matcher.run(success_csv='matches.csv', missing_csv='missing_files.csv', output_dir='match_visualizations')