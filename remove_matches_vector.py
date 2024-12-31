import pandas as pd
import os


def remove_vector(file_path):
    file_name = os.path.basename(file_path)
    new_name = f"dropped_{file_name}"
    new_file_path = f"{os.path.dirname(file_path)}/{new_name}"

    chunks = pd.read_csv(file_path, chunksize=50)
    for i, chunk in enumerate(chunks):
        df = chunk.drop(columns=["matches"])

        if i == 0:
            df.to_csv(new_file_path, mode="w")
        else:
            df.to_csv(new_file_path, mode="a", header=False)


remove_vector("./local_data/sift_matches_v3_w_tp_w_homo_only_1000lines.csv")
