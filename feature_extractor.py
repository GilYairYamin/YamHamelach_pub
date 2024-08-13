import os

from tqdm import tqdm
from PAM_matcher import TwoImagesMatchFeatures, to_patch_fn
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# Access the environment variables
base_path = os.getenv('BASE_PATH')

CSV_IN =  os.path.join(base_path ,  os.getenv('CSV_IN'))
PATCHES_IN = os.path.join(base_path, os.getenv('PATCHES_IN'))
PAIRS_OUT = os.path.join(base_path, os.getenv('PAIRS_OUT'))

# Optional: print to verify
print("CSV_IN:", CSV_IN)
print("PATCHES_IN:", PATCHES_IN)
print("PAIRS_OUT:", PAIRS_OUT)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--patches", help="path for patches" , default=PATCHES_IN)
    parser.add_argument("--pairs_path", help="path for pre-prepared true/false images" , default=PAIRS_OUT)
    parser.add_argument("--features_path", help="path for output true/false features" , default=base_path)
    args = parser.parse_args()

    os.makedirs(args.features_path, exist_ok=True)

    true_list = Path(args.pairs_path, "true", "list.txt")
    false_list = Path(args.pairs_path, "false", "list.txt")

    true_df = pd.read_csv(true_list, names=["piece_A", "piece_B"])
    false_df = pd.read_csv(false_list, names=["piece_A", "piece_B"])
    false_df = false_df.sample(frac=1).reset_index(drop=True)

    features_list = []
    p_bar = tqdm(range(true_df.shape[0]+false_df.shape[0]))
    HEADER = False
    matcher = None
    for label, df in (("true",true_df),("false",false_df)):
        for i,row in df.iterrows():
            p_bar.update(1)
            p_bar.refresh()
            piece_A, piece_B = row.values
            piece_A = to_patch_fn(args.patches, piece_A.split("_")[0], piece_A.split("_")[1])
            piece_B = to_patch_fn(args.patches, piece_B.split("_")[0], piece_B.split("_")[1])
            if matcher is None:
                matcher = TwoImagesMatchFeatures(piece_A, piece_B)
            else:
                matcher.load(piece_A, piece_B)
            matcher.calc_mathces()
            matcher.find_homography()
            matcher.calc_features()
            features, header = matcher.to_features()
            if HEADER is False and header is not False:
                HEADER  = header
                HEADER.extend(['piece_a', 'piece_b', 'label'])

            if features == False:
                print(f"failed to extract features for {piece_A.parts[-1]}, {piece_B.parts[-1]}")
                continue

            features.extend([piece_A.parts[-1], piece_B.parts[-1]])
            features.append(label)
            print(f"successfully extract features for {label}: {piece_A.parts[-1]}, {piece_B.parts[-1]}")

            features_list.append(features)
            if i%100 == 0:
                features_list_df = pd.DataFrame(features_list, columns=HEADER)
                fn = Path(args.features_path, "features.csv")
                features_list_df.to_csv(fn, float_format='%.4f',index=False)
                print(f"saved features into {fn}")
    features_list_df = pd.DataFrame(features_list, columns=HEADER)
    fn = Path(args.features_path, "features.csv")
    features_list_df.to_csv(fn, float_format='%.4f', index=False)
    print(f"saved features into {fn}")

    pass



