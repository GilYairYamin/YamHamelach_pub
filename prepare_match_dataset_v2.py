import os
import csv
import random
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from PAM_matcher import CSVMatcher, FeatureBasedMatcher, save_match_figure
from aux_function import load_pam_files_to_process, print_matched_pam_files

# Set a random seed based on the current time
np.random.seed(datetime.now().microsecond)


def setup_environment():
    """Load environment variables and set up paths."""
    load_dotenv()
    base_path = os.getenv('BASE_PATH')

    CSV_IN = os.path.join(base_path, os.getenv('CSV_IN'))
    PATCHES_IN = os.path.join(base_path, os.getenv('PATCHES_IN'))
    PAIRS_OUT = os.path.join(base_path, os.getenv('PAIRS_OUT'))

    print(f"CSV_IN: {CSV_IN}")
    print(f"PATCHES_IN: {PATCHES_IN}")
    print(f"PAIRS_OUT: {PAIRS_OUT}")

    return CSV_IN, PATCHES_IN, PAIRS_OUT


def print_unique_file_names(csv_file_path):
    """Print unique file names from a CSV file."""
    unique_files = set()

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            file_name = row['File']
            unique_files.add(file_name)

    print("Unique file names:")
    for file_name in sorted(unique_files):
        print(file_name)


def load_pam_files():
    """Load PAM files to process."""
    return load_pam_files_to_process()


def setup_output_directories(pairs_out_path):
    """Set up output directories for matched images."""
    true_dir = Path(pairs_out_path, "true")
    false_dir = Path(pairs_out_path, "false")
    os.makedirs(true_dir, exist_ok=True)
    os.makedirs(false_dir, exist_ok=True)
    return true_dir, false_dir


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description="Process image patches to find matching pairs based on provided CSV and generate match results."
    )
    parser.add_argument(
        "--patches_in",
        type=str,
        default=None,
        help="Path to the directory containing input patches.",
    )
    parser.add_argument(
        "--csv_fn",
        type=str,
        default=None,
        help="Path to the CSV file containing ground truth matches.",
    )
    parser.add_argument(
        "--pairs_out",
        type=str,
        default=None,
        help="Output directory where match results will be saved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--save_figures",
        action="store_true",
        help="Save match figures to output directories.",
    )

    args = parser.parse_args()

    # Load environment variables if arguments are not provided
    if not args.patches_in or not args.csv_fn or not args.pairs_out:
        load_dotenv()
        base_path = os.getenv('BASE_PATH')

        if not args.csv_fn:
            args.csv_fn = os.path.join(base_path, os.getenv('CSV_IN'))
        if not args.patches_in:
            args.patches_in = os.path.join(base_path, os.getenv('PATCHES_IN'))
        if not args.pairs_out:
            args.pairs_out = os.path.join(base_path, os.getenv('PAIRS_OUT'))

    # Validate paths
    if not os.path.isfile(args.csv_fn):
        parser.error(f"The CSV file '{args.csv_fn}' does not exist.")
    if not os.path.isdir(args.patches_in):
        parser.error(f"The patches directory '{args.patches_in}' does not exist.")
    if not os.path.isdir(args.pairs_out):
        os.makedirs(args.pairs_out, exist_ok=True)

    return args


def setup_output_file(pairs_out_path):
    """Set up the output CSV file for match results."""
    results_csv_path = Path(pairs_out_path, "matching_results.csv")
    csvfile = open(results_csv_path, 'w', newline='')
    fieldnames = ['file1', 'file2', 'match_status']
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()
    return csv_writer, csvfile


def get_patch_paths(patches_in_path, pam_list):
    """Collect all patch paths from the given PAM list."""
    patch_paths = []
    for pam in pam_list:
        pam_paths = list(Path(patches_in_path, pam).glob("*.jpg"))
        patch_paths.extend(pam_paths)
    return patch_paths


def process_file_pair(fn_1, fn_2, csv_matcher, csv_writer, save_figures, pairs_out_path):
    """Process a pair of files and write the result."""
    csv_match = csv_matcher.is_match(fn_1, fn_2)
    file_name = f"{fn_1.stem}-{fn_2.stem}.jpg"

    match_status = None
    if csv_match:
        match_status = "true"
    else:
        naive_matcher = FeatureBasedMatcher(fn_1, fn_2, naive=True)
        match = naive_matcher.match()
        if match:
            match_status = "false"

    if match_status:
        # Write the result to CSV
        csv_writer.writerow({
            'file1': fn_1.name,
            'file2': fn_2.name,
            'match_status': match_status
        })

        if save_figures:
            # Save matched images if needed
            output_dir = Path(pairs_out_path, match_status)
            output_path = output_dir / file_name
            save_match_figure(fn_1, fn_2, output_path)


def main():
    """Main function to process image patches."""
    args = parse_arguments()

    if args.verbose:
        print("Arguments:")
        print(f"  Patches In: {args.patches_in}")
        print(f"  CSV File: {args.csv_fn}")
        print(f"  Pairs Out: {args.pairs_out}")

    print("Loading PAM files...")
    print_unique_file_names(args.csv_fn)
    PAM_list = load_pam_files()

    for pam in sorted(PAM_list):
        if args.verbose:
            print(f"Processing PAM: {pam}")

    print_matched_pam_files(args.csv_fn)

    # Prepare output directories
    setup_output_directories(args.pairs_out)

    # Prepare output file and matcher
    csv_writer, csvfile = setup_output_file(args.pairs_out)
    csv_matcher = CSVMatcher(args)

    # Collect all patch paths
    patch_paths = get_patch_paths(args.patches_in, PAM_list)
    random.shuffle(patch_paths)

    with tqdm(total=len(patch_paths)) as pbar:
        for i, fn_1 in enumerate(patch_paths):
            if args.verbose:
                print(f"Processing file {i+1}/{len(patch_paths)}: {fn_1.name}")
            pbar.update(1)
            for fn_2 in patch_paths[i+1:]:
                process_file_pair(
                    fn_1, fn_2, csv_matcher, csv_writer,
                    args.save_figures, args.pairs_out
                )

    csvfile.close()


if __name__ == '__main__':
    main()
