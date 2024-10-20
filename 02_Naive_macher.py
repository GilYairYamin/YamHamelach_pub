import os
import cv2
import pickle
import numpy as np
from typing import List, Dict, Tuple
import csv
from tqdm import tqdm
import itertools


class NaiveImageMatcher:
    def __init__(self, cache_dir: str = "image_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_file_path(self, image_key: str) -> str:
        """Get the file path for a specific image's cache file."""
        return os.path.join(self.cache_dir, f"{image_key}.pkl")

    def _get_image_key(self, file_path: str) -> str:
        """Generate a unique key for the image based on its file path."""
        return os.path.basename(file_path)

    def _serialize_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[Tuple]:
        """Serialize keypoints for saving to the cache."""
        return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    def _deserialize_keypoints(self, keypoints_data: List[Tuple]) -> List[cv2.KeyPoint]:
        """Deserialize keypoints from cached data."""
        return [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1], angle=pt[2],
                             response=pt[3], octave=pt[4], class_id=pt[5]) for pt in keypoints_data]

    def _is_cached(self, image_key: str) -> bool:
        """Check if the cache exists for a specific image."""
        return os.path.exists(self._get_cache_file_path(image_key))

    def _load_cache(self, image_key: str) -> Dict:
        """Load specific cached data for an image from its dedicated cache file."""
        cache_file = self._get_cache_file_path(image_key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, image_key: str, data: Dict):
        """Save the processed image data to a dedicated cache file."""
        cache_file = self._get_cache_file_path(image_key)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def _process_image(self, file_path: str) -> np.ndarray:
        """Process an image, either by loading cached data or computing new descriptors."""
        image_key = self._get_image_key(file_path)

        if self._is_cached(image_key):
            # Load the cached data from its dedicated file
            cached_data = self._load_cache(image_key)
            if cached_data:
                return cached_data['descriptors']

        # If not cached, compute SIFT features and cache the result
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Save the computed data to the dedicated cache file
        self._save_cache(image_key, {
            'keypoints': self._serialize_keypoints(keypoints),
            'descriptors': descriptors
        })

        return descriptors

    def calc_matches(self, file1: str, file2: str) -> List[Tuple[int, int, float]]:
        """Calculate SIFT matches between two images, returning only good matches."""
        des1 = self._process_image(file1)
        des2 = self._process_image(file2)

        try:
            good_matches = []

            bf = cv2.BFMatcher()
            # BFMatcher stands for Brute-Force Matcher. It is used to match feature descriptors
            # between two sets of keypoints from different images. The brute-force approach
            # compares each descriptor from the first set (des1) with all the descriptors from the second set (des2)
            # and returns the best matches based on a chosen distance metric (such as Euclidean distance for SIFT/SURF
            # or Hamming distance for binary descriptors like ORB).

            matches = bf.knnMatch(des1, des2, k=2)

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    # Documentation for m.queryIdx, m.trainIdx, and m.distance:
                    #   • m.queryIdx: The index of the keypoint from des1 (the query descriptor set).
                    #   • m.trainIdx: The index of the matched keypoint from des2 (the training descriptor set).
                    #   • m.distance: The distance between the two descriptors. This represents how similar
                    #   the two descriptors are. A lower distance indicates a better match.
                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))
        except Exception as e:
            print(f"Error occurred while filtering matches: {e}")
            return []

        return good_matches


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

    def calculate_distances(self, image_files: List[str], success_csv: str , debug: bool = False) -> None:
        total_iterations = sum(range(1, len(image_files)))  # Total number of comparisons

        if debug == True:
            import tracemalloc
            import objgraph
            import psutil

            tracemalloc.start()



        with open(success_csv, mode='w', newline='') as file:
            fieldnames = ['file1', 'file2', 'distance', 'matches']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            with tqdm(total=total_iterations, desc="Processing Patches", disable=False) as pbar:
                for i, j in itertools.combinations(range(len(image_files)), 2):
                    if (debug == True) and ((i % 2 == 0) and (j == i+1)):
                        os.system('clear')
                        print(f"====== {i} , {j}")

                        # Get current process
                        process = psutil.Process(os.getpid())

                        # CPU usage in percentage
                        cpu_usage = process.cpu_percent(interval=1)
                        print(f"CPU Usage: {cpu_usage}%")

                        # Memory usage in MB
                        memory_info = process.memory_info()
                        print(f"Memory Usage: {memory_info.rss / 1024 ** 2:.2f} MB")

                        # Visualize the most common types of objects
                        objgraph.show_most_common_types()

                        # Find out which objects are consuming memory
                        objgraph.show_growth()

                        # Get memory snapshot
                        snapshot = tracemalloc.take_snapshot()

                        # Display the top memory-consuming lines
                        top_stats = snapshot.statistics('lineno')
                        print("[ Top 10 Memory Consuming Variables ]")
                        for stat in top_stats[:10]:
                            print(stat)

                    # Inner loop for calculating matches
                    pbar.update(1)
                    image_path1 = image_files[i]
                    image_path2 = image_files[j]
                    good_matches = self.matcher.calc_matches(image_path1, image_path2)
                    # Write match details to the CSV
                    writer.writerow({
                        'file1': image_path1,
                        'file2': image_path2,
                        'distance': len(good_matches),
                        'matches': [(m[0], m[1], m[2]) for m in good_matches]
                    })

    def run(self, success_csv='matches_v3.csv' , debug=False):
        image_files = self.get_image_files()
        self.calculate_distances(image_files, success_csv , debug=debug)
        print(f"Results written to {success_csv}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DEBUG = os.getenv('DEBUG')
    base_path = os.getenv('BASE_PATH')
    PATCHES_DIR = os.path.join(base_path, os.getenv('PATCHES_IN'))
    matcher = FragmentMatcher(PATCHES_DIR)
    matcher.run(success_csv='matches_v3.csv' , debug=DEBUG)