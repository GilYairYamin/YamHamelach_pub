import csv
import itertools
import os
import pickle
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from env_arguments_loader import load_env_arguments

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)


# Cache manager for storing image descriptors and keypoints
class DescriptorCacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_file_path(self, image_key: str) -> str:
        """Generate the file path for a specific image's cache file."""
        return os.path.join(self.cache_dir, f"{image_key}.pkl")

    def _is_cached(self, image_key: str) -> bool:
        """Check if the cache exists for a specific image."""
        return os.path.exists(self._get_cache_file_path(image_key))

    def _load_cache(self, image_key: str) -> Dict:
        """Load cached data for an image."""
        cache_file = self._get_cache_file_path(image_key)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, image_key: str, data: Dict):
        """Save the processed image data to a cache file."""
        cache_file = self._get_cache_file_path(image_key)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    def _serialize_keypoints(
        self, keypoints: List[cv2.KeyPoint]
    ) -> List[Tuple]:
        """Serialize keypoints for saving to the cache."""
        return [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints
        ]

    def _deserialize_keypoints(
        self, keypoints_data: List[Tuple]
    ) -> List[cv2.KeyPoint]:
        """Deserialize keypoints from cached data."""
        return [
            cv2.KeyPoint(
                x=pt[0][0],
                y=pt[0][1],
                size=pt[1],
                angle=pt[2],
                response=pt[3],
                octave=pt[4],
                class_id=pt[5],
            )
            for pt in keypoints_data
        ]

    def process_image(
        self, file_path: str
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Process an image, either by loading cached data or computing new descriptors."""
        image_key = os.path.basename(file_path)

        if self._is_cached(image_key):
            # Load cached data if available
            cached_data = self._load_cache(image_key)
            if cached_data:
                keypoints = self._deserialize_keypoints(
                    cached_data["keypoints"]
                )
                descriptors = cached_data["descriptors"]
                return keypoints, descriptors

        # If not cached, compute SIFT features and cache the result
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Save the computed data to the cache
        self._save_cache(
            image_key,
            {
                "keypoints": self._serialize_keypoints(keypoints),
                "descriptors": descriptors,
            },
        )

        return keypoints, descriptors


# Class to handle matching between images using SIFT and BFMatcher
class NaiveImageMatcher:
    def __init__(self, descriptor_cache: DescriptorCacheManager):
        self.descriptor_cache = descriptor_cache

    def calc_matches(self, file1: str, file2: str) -> List[cv2.DMatch]:
        """Calculate SIFT matches between two images, returning only good matches."""
        # Get descriptors and keypoints for both images
        kp1, des1 = self.descriptor_cache.process_image(file1)
        kp2, des2 = self.descriptor_cache.process_image(file2)

        try:
            good_matches = []
            bf = cv2.BFMatcher()
            # BFMatcher stands for Brute-Force Matcher. It compares each descriptor
            # from des1 with all the descriptors from des2.
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test to filter out good matches
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        except Exception as e:
            print(f"Error occurred while filtering matches: {e}")
            return []

        return good_matches


class FragmentMatcher:
    def __init__(self, image_base_path: str, cash_dir: str):
        self.image_base_path = image_base_path
        self.matcher = NaiveImageMatcher(DescriptorCacheManager(cash_dir))

    def get_image_files(self) -> List[str]:
        """Get a list of all image file paths in the image_base_path directory."""

        image_files = []
        for root, _, files in os.walk(self.image_base_path):
            for file in files:
                if file.endswith(
                    ".jpg"
                ):  # Assuming patches are in .jpg format
                    image_files.append(os.path.join(root, file))
        return image_files

    def _get_processed_pairs(self, success_csv: str) -> set:
        """Read the CSV file and get the set of already processed image pairs."""
        processed_pairs = set()
        if os.path.exists(success_csv):
            with open(success_csv, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    processed_pairs.add((row["file1"], row["file2"]))
        return processed_pairs

    def calculate_distances(
        self, image_files: List[str], success_csv: str, debug: bool = False
    ) -> None:
        total_iterations = sum(
            range(1, len(image_files))
        )  # Total number of comparisons
        processed_pairs = self._get_processed_pairs(success_csv)

        with open(success_csv, mode="a", newline="") as file:
            fieldnames = ["file1", "file2", "distance", "matches"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header only if the file is newly created
            if not processed_pairs:
                writer.writeheader()

            with tqdm(
                total=total_iterations,
                desc="Processing Patches",
                disable=False,
            ) as pbar:
                for i, j in itertools.combinations(range(len(image_files)), 2):
                    image_path1 = image_files[i]
                    image_path2 = image_files[j]

                    dirname1 = os.path.dirname(image_path1)
                    dirname2 = os.path.dirname(image_path2)

                    base_name1 = os.path.basename(image_path1)
                    base_name2 = os.path.basename(image_path2)

                    # different patches can't be from the same directory
                    if dirname1 == dirname2:
                        pbar.update(1)  # Update progress bar
                        continue
                    # Skip if the images are from the same directory

                    # Check if the pair has already been processed
                    if (image_path1, image_path2) in processed_pairs or (
                        image_path2,
                        image_path1,
                    ) in processed_pairs:
                        pbar.update(1)  # Update progress bar
                        continue  # Skip this pair

                    # Inner loop for calculating matches
                    pbar.update(1)
                    good_matches = self.matcher.calc_matches(
                        image_path1, image_path2
                    )

                    if len(good_matches) <= 0:
                        continue
                    # Write match details to the CSV
                    writer.writerow(
                        {
                            "file1": base_name1,
                            "file2": base_name2,
                            "distance": len(good_matches),
                            "matches": [
                                (m.queryIdx, m.trainIdx, m.distance)
                                for m in good_matches
                            ],
                        }
                    )

                    # Flush to ensure data is written to the file immediately
                    file.flush()

    def run(self, success_csv, debug=False):
        image_files = self.get_image_files()
        self.calculate_distances(image_files, success_csv, debug=debug)
        print(f"Results written to {success_csv}")


if __name__ == "__main__":
    args = load_env_arguments()

    patches_dir = os.path.join(args.base_path, args.patches_in)
    patche_cache_dir = os.path.join(args.base_path, args.patches_cache)

    matcher = FragmentMatcher(patches_dir, patche_cache_dir)

    _sift_matches = os.path.join(args.base_path, args.sift_matches)
    matcher.run(success_csv=_sift_matches, debug=args.debug)
