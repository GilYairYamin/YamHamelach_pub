import os

import pandas as pd

# ERROR = 40
CHUNK_SIZE = 100000


def clean_csv(file_path):
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

        chunk.dropna(inplace=True)
        chunk.drop(columns=["matches", "Match", "is_valid"], inplace=True)

        if start is True:
            start = False
            chunk.to_csv(new_file_path, mode="w", index=False)
            continue

        chunk.to_csv(new_file_path, mode="a", header=False, index=False)

    df = pd.read_csv(new_file_path, low_memory=False)
    df.sort_values(by=["file1", "file2"], inplace=True)
    df.to_csv(new_file_path, mode="w", index=False)
    return new_file_path


def generate_general_matches(file_path):
    file_name = os.path.basename(file_path)
    file_dir_name = os.path.dirname(file_path)
    new_file_name = f"general_{file_name}"
    new_file_path = os.path.join(file_dir_name, new_file_name)
    new_files_dir_name = os.path.join(file_dir_name, "./matches_csvs")

    os.makedirs(new_files_dir_name, exist_ok=True)
    result_dict = {}
    # Process the file in chunks
    i = 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        i += 1
        print(f"Processing chunk {i}, with shape: {chunk.shape}")
        for idx, row in chunk.iterrows():
            arr = [row["file1"], row["file2"]]
            arr.sort()

            key = f"{arr[0]}_{arr[1]}.csv"
            curr_file_path = os.path.join(new_files_dir_name, key)
            row_df = row.to_frame().T
            row_df.drop(columns=["file1", "file2"], inplace=True)

            if key not in result_dict.keys():
                result_dict[key] = arr
                row_df.to_csv(curr_file_path, mode="w", index=False)
                continue

            row_df.to_csv(curr_file_path, mode="a", index=False, header=False)

    new_data = [["file1", "file2"]]
    for value in result_dict.values():
        new_data.append(value)

    new_data.sort(key=lambda obj: f"{obj[0]}_{obj[1]}")

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(new_file_path, mode="w")
    return new_file_path


clean_csv("./local_data/sift_matches_v3_w_tp_w_homo.csv")
