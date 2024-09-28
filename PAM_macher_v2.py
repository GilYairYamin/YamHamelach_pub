import cv2
import pickle
import os
from tqdm import tqdm  # Progress bar
import numpy as np

import cv2
import pickle
import os
from typing import Dict, List, Tuple


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
    def __init__(self, pam_csv_file, image_base_path):
        self.pam_csv_file = pam_csv_file
        self.image_base_path = image_base_path
        self.matcher = NaiveImageMatcher()
        self.missing_files = []  # To store missing files

    def read_csv(self):
        """Read the CSV file and return it as a pandas DataFrame."""
        return pd.read_csv(self.pam_csv_file)

    def get_image_path(self, file_name, box):
        """Generate the image path based on file and box information."""

        image_file = f"{file_name}_{int(box)}.jpg"
        return os.path.join(self.image_base_path, file_name, image_file)

    def print_matched_pam_files(self):
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
        """Go over the folders and calculate distances between matching images."""
        distances = []

        for index, row in tqdm(matches_df.iterrows(), total=matches_df.shape[0], desc="Processing Matches"):
            image_path1 = self.get_image_path(row['File_x'], row['Box_x'])
            image_path2 = self.get_image_path(row['File_y'], row['Box_y'])

            if os.path.exists(image_path1) and os.path.exists(image_path2):
                distance = self.matcher.calc_matches(image_path1, image_path2)
                distances.append({
                    'scroll': row['Scroll'],
                    'fragment': row['Frg'],
                    'file1': image_path1,
                    'file2': image_path2,
                    'distance': len(distance[4])  # Number of good matches
                })
            else:
                # Store missing file details for logging
                self.missing_files.append({
                    'scroll': row['Scroll'],
                    'fragment': row['Frg'],
                    'file1': image_path1,
                    'file2': image_path2
                })

        return distances

    def write_csv(self, distances, success_csv, missing_csv):
        """Write the results to two CSV files."""
        # Write the successful match results to CSV
        with open(success_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['scroll', 'fragment', 'file1', 'file2', 'distance'])
            writer.writeheader()
            writer.writerows(distances)

        # Write the missing file information to another CSV
        with open(missing_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['scroll', 'fragment', 'file1', 'file2'])
            writer.writeheader()
            writer.writerows(self.missing_files)

    def run(self, success_csv='matches.csv', missing_csv='missing_files.csv'):
        """Main method to run the matcher, calculate distances, and write to CSV."""
        matches_df = self.print_matched_pam_files()
        distances = self.calculate_distances(matches_df)
        self.write_csv(distances, success_csv, missing_csv)
        print(f"Results written to {success_csv} and missing file info to {missing_csv}")


if __name__ == "__main__":
    # Usage
    csv_file = "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/PAM_Identification_for_Asaf_Round_1_avinoamEdit.csv"
    image_base_path = "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/bounding_boxes_crops/patches/"
    matcher = FragmentMatcher(csv_file, image_base_path)
    matcher.run(success_csv='matches.csv', missing_csv='missing_files.csv')
