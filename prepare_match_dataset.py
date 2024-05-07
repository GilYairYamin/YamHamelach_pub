import os
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import numpy as np

from src.PAM_matcher import CSVMatcher, FeatureBasedMatcher, save_match_figure

np.random.seed(datetime.now().microsecond)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--patches")
    # csv_fn /home/avinoam/workspace/YAM_HAMELACH/results/17_05/PAM_Identification_for_Asaf_Round_1.csv
    parser.add_argument("--csv_fn", default=False)
    parser.add_argument("--pairs_out", default=False)
    args = parser.parse_args()

    # prepare output locations
    os.makedirs(Path(args.pairs_out, "true"), exist_ok=True)
    os.makedirs(Path(args.pairs_out, "false"), exist_ok=True)
    true_fn = Path(args.pairs_out, "true","list.txt")
    false_fn = Path(args.pairs_out, "false","list.txt")
    true_file = open(true_fn,'w')
    false_file = open(false_fn,'w')


    # read ground true
    csv_matcher = CSVMatcher(args)

    patch_path = []
    for PAM in ['M43485-1-E', 'M43003-1-E', 'M43510-1-E',
                'M43500-1-E', 'M43505-1-E', 'M42809-1-E',
                'M43473-1-E', 'M42970-1-E', 'M43992-1-E',
                'M43195-1-E', 'M43448-1-E']:
        PAM_paths = list(Path(args.patches,PAM).glob("*.jpg"))
        patch_path.extend(PAM_paths)


    p_bar = tqdm(np.random.shuffle(patch_path))

    for i,fn_1 in enumerate(patch_path):
        p_bar.update(1)
        p_bar.refresh()
        for (j,fn_2) in enumerate(patch_path[i+1:]):
            csv_match = csv_matcher.is_match(fn_1, fn_2)


            if csv_match:
                print(f"{fn_1.name},{fn_2.name} true detection according to csv ")
                true_file.write(f"{fn_1.name},{fn_2.name}{os.linesep}")
                # missed.write(f"f{fn_1.name},{fn_2.name},{os.linesep}")
                path = Path(args.pairs_out, "true",
                            f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
                # save_match_figure(fn_1, fn_2, path)

            else:
                if np.random.random()>0.1:
                    continue
                naive_matcher = FeatureBasedMatcher(fn_1, fn_2, naive=True)
                # match = naive_matcher.match() # naive matching, as was set in init mode
                match = True
                if match:
                    false_file.write(f"{fn_1.name},{fn_2.name}{os.linesep}")
                    print (f"false match detected for {fn_1.name}, {fn_2.name}, saving to dataset")
                    # features, features_names = naive_matcher.features()
                    path = Path(args.pairs_out, "false",
                                  f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
                    # save_match_figure(fn_1, fn_2, path)
        false_file.flush()
        true_file.flush()

    false_file.close()
    true_file.close()
