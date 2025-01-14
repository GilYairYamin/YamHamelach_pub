import csv
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from PAM_matcher import CSVMatcher, FeatureBasedMatcher, save_match_figure

np.random.seed(datetime.now().microsecond)


# Load the .env file
load_dotenv()

# Access the environment variables
base_path = os.getenv("BASE_PATH")

CSV_IN = os.path.join(base_path, os.getenv("CSV_IN"))
PATCHES_IN = os.path.join(base_path, os.getenv("PATCHES_IN"))
PAIRS_OUT = os.path.join(base_path, os.getenv("PAIRS_OUT"))

# Optional: print to verify
print("CSV_IN:", CSV_IN)
print("PATCHES_IN:", PATCHES_IN)
print("PAIRS_OUT:", PAIRS_OUT)

eee = 1


def print_unique_file_names(csv_file_path):
    unique_files = set()

    with open(csv_file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            file_name = row["\ufeffFile"]
            unique_files.add(file_name)
    sorted_unique_files = sorted(unique_files)
    print("Unique file names:")
    for file_name in sorted_unique_files:
        print(file_name)


from aux_function import load_pam_files_to_process, print_matched_pam_files

if __name__ == "__main__":
    print("Loading PAM files...")
    print("print stats...")

    print_unique_file_names(CSV_IN)
    PAM_list = load_pam_files_to_process()

    sorted_PAM_list = sorted(PAM_list)
    for p in sorted_PAM_list:
        print(f"processing PAM: {p}")

    print_matched_pam_files(CSV_IN)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--patches_in", default=PATCHES_IN, type=str)
    # csv_fn /home/avinoam/workspace/YAM_HAMELACH/results/17_05/PAM_Identification_for_Asaf_Round_1.csv
    parser.add_argument("--csv_fn", default=CSV_IN)
    parser.add_argument("--pairs_out", default=PAIRS_OUT, type=str)
    args = parser.parse_args()

    # prepare output locations
    os.makedirs(Path(args.pairs_out, "true"), exist_ok=True)
    os.makedirs(Path(args.pairs_out, "false"), exist_ok=True)
    true_fn = Path(args.pairs_out, "true", "list.txt")
    false_fn = Path(args.pairs_out, "false", "list.txt")
    true_file = open(true_fn, "w")
    false_file = open(false_fn, "w")

    # read ground true
    csv_matcher = CSVMatcher(args)

    patch_path = []
    for PAM in PAM_list:
        PAM_paths = list(Path(args.patches_in, PAM).glob("*.jpg"))
        patch_path.extend(PAM_paths)

    p_bar = tqdm(np.random.shuffle(patch_path))

    for i, fn_1 in enumerate(patch_path):
        print(f"process file {i + 1}/{len(patch_path)}: {fn_1.name}")

        p_bar.update(1)
        p_bar.refresh()
        for j, fn_2 in enumerate(patch_path[i + 1 :]):
            csv_match = csv_matcher.is_match(fn_1, fn_2)

            file_name = f"{fn_1.stem}-{fn_2.stem}.jpg"

            sub_path = None
            if csv_match:
                sub_path = "true"
                true_file.write(file_name)
            else:
                naive_matcher = FeatureBasedMatcher(fn_1, fn_2, naive=True)
                match = (
                    naive_matcher.match()
                )  # naive matching, as was set in init mode
                # match = True
                if match:
                    sub_path = "false"
                    false_file.write(file_name)

            if sub_path is not None:
                path = Path(
                    args.pairs_out, sub_path, f"{fn_1.stem}-{fn_2.stem}.jpg"
                )
                # save_match_figure(fn_1, fn_2, path)

            # if csv_match:
            #     print(f"{fn_1.name},{fn_2.name} true detection according to csv ")
            #     true_file.write(f"{fn_1.name},{fn_2.name}{os.linesep}")
            #     # missed.write(f"f{fn_1.name},{fn_2.name},{os.linesep}")
            #     path = Path(args.pairs_out, "true",
            #                 f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
            #     # save_match_figure(fn_1, fn_2, path)
            #
            # else:
            #     if np.random.random()>0.1:
            #         continue
            #     #naive_matcher = FeatureBasedMatcher(fn_1, fn_2, naive=True)
            #     # match = naive_matcher.match() # naive matching, as was set in init mode
            #     match = True
            #     if match:
            #         false_file.write(f"{fn_1.name},{fn_2.name}{os.linesep}")
            #         print (f"false match detected for {fn_1.name}, {fn_2.name}, saving to dataset")
            #         # features, features_names = naive_matcher.features()
            #         path = Path(args.pairs_out, "false",
            #                       f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
            # save_match_figure(fn_1, fn_2, path)
        false_file.flush()
        true_file.flush()

    false_file.close()
    true_file.close()
