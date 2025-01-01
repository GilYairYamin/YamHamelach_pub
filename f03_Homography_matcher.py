import csv
import os
import pickle
import sys
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from f02_SIFT_macher import DescriptorCacheManager, NaiveImageMatcher

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)


# Cache manager for storing homography error arrays
class ErrorCacheManager:
    def __init__(self, cache_dir: str = "error_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_file_path(self, file1: str, file2: str) -> str:
        """Generate a cache file path for the image pair."""
        pair_key = f"{os.path.basename(file1)}_{os.path.basename(file2)}.pkl"
        return os.path.join(self.cache_dir, pair_key)

    def _is_cached(self, file1: str, file2: str) -> bool:
        """Check if the error cache exists for a specific image pair."""
        return os.path.exists(self._get_cache_file_path(file1, file2))

    def _load_cache(self, file1: str, file2: str) -> np.ndarray:
        """Load the cached error array for a specific image pair."""
        cache_file = self._get_cache_file_path(file1, file2)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, file1: str, file2: str, errors: np.ndarray):
        """Save the error array for a specific image pair."""
        cache_file = self._get_cache_file_path(file1, file2)
        with open(cache_file, "wb") as f:
            pickle.dump(errors, f)


# Homography and error calculation class
class HomographyErrorCalculator:
    def __init__(
        self,
        matcher: NaiveImageMatcher,
        error_cache: ErrorCacheManager,
        descriptor_cache: DescriptorCacheManager,
    ):
        self.matcher = matcher
        self.error_cache = error_cache
        self.descriptor_cache = descriptor_cache

    def _compute_homography_and_errors(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the homography matrix and the error array for a set of good matches."""
        ptsA = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype="float")
        ptsB = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype="float")

        # Compute the homography matrix using RANSAC
        if ptsA.shape[0] < 4 or ptsB.shape[0] < 4:
            return [], []

        H, _ = cv2.findHomography(ptsB, ptsA, method=cv2.RANSAC)
        if H is None:
            return [], []

        # Compute projection errors
        errors = np.zeros(len(good_matches), dtype="float")
        for i, m in enumerate(good_matches):
            pt_b_homogenous = np.concatenate(
                (ptsB[i], np.array([1]))
            )  # Convert to homogenous coordinates
            projected_p = H.dot(pt_b_homogenous)
            projected_p_homogenous = (
                projected_p / projected_p[2]
            )  # Normalize by the third (homogeneous) component
            projected_p = projected_p_homogenous[:2]
            error = np.linalg.norm(projected_p - ptsA[i])
            errors[i] = error

        return H, errors

    def calculate_errors_for_pair(
        self, file1: str, file2: str, good_matches_count: int
    ) -> np.ndarray:
        """Calculate or load cached error array for an image pair."""
        # Check if the errors are already cached
        if self.error_cache._is_cached(file1, file2):
            return self.error_cache._load_cache(file1, file2)

        errors = []
        # Calculate matches and keypoints
        good_matches = self.matcher.calc_matches(file1, file2)[:good_matches_count]
        if not good_matches:
            return errors
            # raise ValueError(f"No good matches found between {file1} and {file2}")

        kp1, des1 = self.descriptor_cache.process_image(file1)
        kp2, des2 = self.descriptor_cache.process_image(file2)

        # Compute the homography and errors
        _, errors = self._compute_homography_and_errors(kp1, kp2, good_matches)
        if len(errors) > 0:
            # Save the error array to cache
            self.error_cache._save_cache(file1, file2, errors)

        return errors


# Function to read image pairs and good matches count from a CSV file with yield


def read_image_pairs_from_csv(csv_file: str):
    """Read image pairs and the number of good matches from a CSV file, yielding each row."""
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 3:
                file1 = row[0]
                file2 = row[1]
                good_matches_count = int(
                    row[2]
                )  # Get the number of good matches from the CSV
                yield file1, file2, good_matches_count


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DEBUG = os.getenv("DEBUG")
    base_path = os.getenv("BASE_PATH")

    _sift_matches = os.path.join(base_path, os.getenv("SIFT_MATCHES"))
    if not os.path.exists(_sift_matches):
        raise FileNotFoundError(f"SIFT matches file not found at {_sift_matches}")

    image_cache_dir = os.path.join(base_path, os.getenv("PATCHES_CACHE"))

    descriptor_cache = DescriptorCacheManager(image_cache_dir)
    matcher = NaiveImageMatcher(descriptor_cache)
    HOMOGRAPHY_CACHE_DIR = os.path.join(base_path, os.getenv("HOMOGRAPHY_CACHE"))
    error_cache = ErrorCacheManager(HOMOGRAPHY_CACHE_DIR)
    error_calculator = HomographyErrorCalculator(matcher, error_cache, descriptor_cache)

    # Read image pairs and good matches from CSV file using generator
    for file1, file2, good_matches_count in tqdm(
        read_image_pairs_from_csv(_sift_matches), desc="Calculating Errors"
    ):
        try:
            errors = error_calculator.calculate_errors_for_pair(
                file1, file2, good_matches_count
            )
            # print(f"Error array for {file1} and {file2}: {errors}")
        except ValueError as e:
            print(e)
