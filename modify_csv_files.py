import os

import pandas as pd

# ERROR = 40
CHUNK_SIZE = 100000


def clean_csv(
    file_path,
    error_metric: str = None,
    error_threshold: float = None,
    distance_threshold: float = None,
):
    file_name = os.path.basename(file_path)
    file_dir_name = os.path.dirname(file_path)
    new_file_name = f"clean_{file_name}"
    new_file_path = os.path.join(file_dir_name, new_file_name)

    start = True
    # Process the file in chunks
    i = 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        i += 1
        print(f"Processing chunk {i}, with shape: {chunk.shape}")

        chunk.drop(columns=["matches", "Match", "is_valid"], inplace=True)
        chunk.dropna(inplace=True)

        if error_metric is not None and error_threshold is not None:
            chunk = chunk[chunk[error_metric] <= error_threshold]

        if distance_threshold is not None:
            chunk = chunk[chunk["distance"] >= distance_threshold]

        if start is True:
            start = False
            chunk.to_csv(new_file_path, mode="w", index=False)
            continue

        chunk.to_csv(new_file_path, mode="a", header=False, index=False)

    df = pd.read_csv(new_file_path, low_memory=False)
    df.sort_values(by=["file1", "file2"], inplace=True)
    df.to_csv(new_file_path, mode="w", index=False)
    return new_file_path


clean_csv("./local_data/sift_matches_v3_w_tp_w_homo.csv")
