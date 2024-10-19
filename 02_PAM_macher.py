import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class NaiveImageMatcher:
    def __init__(self, cache_file: str = "image_cache_v3.pkl"):
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
            #print(f"Loading cached data for {file_path}")
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
        # print(">>>>>>>>>>>>>>>>> kp1")
        # print(len(kp1))
        # print(kp1)
        # print(">>>>>>>>>>>>>>>>> kp2")
        # print(len(kp2))
        # print(kp2)
        # print(">>>>>>>>>>>>>>>>> des1")
        # print(len(des1))
        # print(des1)
        # print(">>>>>>>>>>>>>>>>> des2")
        # print(len(des2))
        # print(des2)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        #print(f"Found {len(matches)} matches between {file1} and {file2}")

        good_matches = []
        print(f"Filtering {len(matches)} good matches {matches}")
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))
        except Exception as e:
            print(f"Error occurred while filtering matches: {e}")


        return kp1, kp2, des1, des2, good_matches


class FragmentMatcher:
    def __init__(self, image_base_path):
        self.image_base_path = image_base_path
        self.matcher = NaiveImageMatcher()

    def get_image_files(self) -> List[str]:
        """Get a list of all image file paths in the image_base_path directory."""
        image_files = []
        for root, _, files in os.walk(self.image_base_path):
            for file in files:
                if file.endswith(".jpg"):  # Assuming patches are in .jpg format
                    image_files.append(os.path.join(root, file))
        return image_files

    def calculate_distances(self, image_files: List[str]):
        distances = []

        for i in tqdm(range(len(image_files)), desc="Processing Patches"):
            for j in range(i + 1, len(image_files)):  # Avoid duplicate comparisons
                image_path1 = image_files[i]
                image_path2 = image_files[j]
                print(f"Comparing {image_path1} with {image_path2}")
                kp1, kp2, des1, des2, good_matches = self.matcher.calc_matches(image_path1, image_path2)
                distances.append({
                    'file1': image_path1,
                    'file2': image_path2,
                    'distance': len(good_matches),
                    'keypoints1': [(kp.pt[0], kp.pt[1]) for kp in kp1],
                    'keypoints2': [(kp.pt[0], kp.pt[1]) for kp in kp2],
                    'matches': [(m[0], m[1]) for m in good_matches]
                })

        return distances

    def write_results(self, distances, success_csv):
        """Write the results to a CSV file."""
        with open(success_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['file1', 'file2', 'distance'])
            writer.writeheader()
            for match in distances:
                writer.writerow({
                    'file1': match['file1'],
                    'file2': match['file2'],
                    'distance': match['distance']
                })

    def run(self, success_csv='matches.csv'):
        image_files = self.get_image_files()
        distances = self.calculate_distances(image_files)
        self.write_results(distances, success_csv)
        print(f"Results written to {success_csv}")


from dotenv import load_dotenv
import os


load_dotenv()
base_path = os.getenv('BASE_PATH')


PATCHES_DIR = os.path.join(base_path, os.getenv('PATCHES_IN'))

if __name__ == "__main__":
    matcher = FragmentMatcher(PATCHES_DIR)
    matcher.run(success_csv='matches_v3.csv')
