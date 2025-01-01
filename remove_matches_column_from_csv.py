import pandas as pd
import os

ERROR = 40
CHUNK_SIZE = 10000


def clean_csv(file_path):
    file_name = os.path.basename(file_path)
    new_file_name = f"clean_{file_name}"
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    start = True
    # Process the file in chunks
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        print(f"Processing chunk with shape: {chunk.shape}")

        chunk.drop(columns=["matches"], inplace=True)
        chunk = chunk[chunk["mean_homo_err"] <= ERROR]

        if start is True:
            start = False
            chunk.to_csv(new_file_path, mode="w")
            continue

        chunk.to_csv(new_file_path, mode="a", header=False)


clean_csv("./local_data/sift_matches_v3_w_tp_w_homo.csv")
