# aux.py

import os

import pandas as pd
from dotenv import load_dotenv


def load_pam_files_to_process():
    """
    Loads a comma-separated string from an environment variable and returns it as a list.

    :param var_name: The name of the environment variable to load.
    :return: A list of strings.
    """
    var_name = "PAM_FILES_TO_PROCESS"

    # Load the .env file
    load_dotenv()

    # Retrieve the string from the environment variable
    array_var = os.getenv(var_name)

    # Split the string into a list, if it exists
    if array_var:
        return array_var.split(",")
    else:
        return []


def print_matched_pam_files(pam_csv_file):
    # Perform an inner join on the Scroll and Frg columns

    df = pd.read_csv(pam_csv_file)

    # Remove rows where Box is NaN
    df = df.dropna(subset=["Box"])

    merged_df = pd.merge(df, df, on=["Scroll", "Frg"])

    # Filter the results where the Box values are different
    filtered_df = merged_df[merged_df["Box_x"] != merged_df["Box_y"]]

    # Select only the necessary columns: file1, box1, file2, box2
    result_df = filtered_df[["File_x", "Box_x", "File_y", "Box_y"]]
    print(result_df.to_string(index=False))
