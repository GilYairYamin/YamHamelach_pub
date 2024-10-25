import pandas as pd

# Define the file paths
input_file =  "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/matches_v3_with_tp.csv"
output_file =  "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/matches_v3_with_tp_sort.csv"

# Read the CSV file
df = pd.read_csv(input_file)

df_sorted_desc = df.sort_values(by=df.columns[2], ascending=False)
# Sort by the third column (index 2, since index starts at 0)

# Select the first 1000 rows
df_sorted_first_1000 = df_sorted_desc.head(200)

# Write the first 1000 rows to a new CSV file
df_sorted_first_1000.to_csv(output_file)

print("File saved to:", output_file)