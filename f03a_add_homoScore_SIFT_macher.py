import pandas as pd
import os
import pickle
from dotenv import load_dotenv
from tqdm import tqdm
import math

load_dotenv()
base_path = os.getenv('BASE_PATH')

# in csv
csv_sift_matches_w_tp = os.path.join(base_path,os.getenv('SIFT_MATCHES_W_TP'))

# out csv
csv_sift_matches_w_tp_w_homo = os.path.join(base_path, os.getenv('SIFT_MATCHES_W_TP_W_HOMO'))

# cash dir...
pkl_directory = os.path.join(base_path, os.getenv('HOMOGRAPHY_CACHE'))


# Load CSV file
df = pd.read_csv(csv_sift_matches_w_tp)

# Initialize a new column to store the averages
df['sum_homo_err'] = -1
df['len_homo_err'] = -1

# Iterate over each row in the CSV file
#for index, row in df.iterrows():
for index, row in tqdm(df.iterrows(), total=len(df)):

    file1, file2 = row['file1'], row['file2']

    # Generate the corresponding .pkl filename
    pkl_filename = f"{file1}_{file2}.pkl"
    pkl_path = os.path.join(pkl_directory, pkl_filename)

    # Check if the .pkl file exists
    if os.path.exists(pkl_path):
        # Load the .pkl file and calculate the average
        with (open(pkl_path, 'rb') as pkl_file):
            try:
                pkl_content = pickle.load(pkl_file)

                # Convert pickle content to a list (if it's not already)
                data = list(pkl_content)

                # Calculate sum, length, and mean
                sum_value = sum(data)
                len_value = len(data)

                # Calculate mean
                mean_value = sum_value / len_value if len_value > 0 else 0

                # Calculate the standard deviation
                if len_value > 1:
                    sum_of_squared_diff = sum((x - mean_value) ** 2 for x in data)
                    std_dev = math.sqrt(sum_of_squared_diff / (len_value - 1))  # Sample standard deviation
                else:
                    std_dev = 0  # In case there's only one item, the std deviation is 0

                # Add values to the DataFrame
                df.at[index, 'sum_homo_err'] = sum_value
                df.at[index, 'len_homo_err'] = len_value
                df.at[index, 'mean_homo_err'] = mean_value
                df.at[index, 'std_homo_err'] = std_dev
                df.at[index, 'is_valid'] = True

            except Exception as e:
                print(f"Error reading {pkl_filename}: {e}")
    else:
        print(f".pkl file not found: {pkl_filename}")

# Filter only the valid rows
valid_df = df[df['is_valid'] | df['Match']]

# Save the updated CSV file
valid_df.to_csv(csv_sift_matches_w_tp_w_homo, index=False)

# Save the updated CSV file
df.to_csv(csv_sift_matches_w_tp_w_homo, index=False)

print(f"Processing completed. The updated CSV file has been saved as {csv_sift_matches_w_tp_w_homo}.")
