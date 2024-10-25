import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


class PamProcessor:
    def __init__(self, image_base_path, csv_filename, _sift_matches):
        self.pam_csv_file = csv_filename
        assert os.path.exists(self.pam_csv_file), f"PAM CSV file not found: {self.pam_csv_file}"

        self.image_base_path = image_base_path
        assert os.path.exists(self.image_base_path), f"Image base path not found: {self.image_base_path}"

        self._sift_matches = _sift_matches
        assert os.path.exists(self._sift_matches), f"SIFT matches file not found: {self._sift_matches}"

    def read_csv(self):
        """Read the CSV file in chunks to avoid loading it all into memory."""
        return pd.read_csv(self.pam_csv_file)

    def read_csv_chunks(self, chunk_size=100000):
        """Read the CSV file in chunks to avoid loading it all into memory."""
        return pd.read_csv(self._sift_matches, chunksize=chunk_size)

    def get_matched_pam_files(self):
        """Find matches based on Scroll and Fragment with different boxes."""
        df = self.read_csv()

        # Remove rows where Box is NaN
        df = df.dropna(subset=['Box'])

        # Perform an inner join on the Scroll and Frg columns
        merged_df = pd.merge(df, df, on=['Scroll', 'Frg'])

        # Filter the results where the Box values are different
        filtered_df = merged_df[merged_df['Box_x'] != merged_df['Box_y']]

        # Select only the necessary columns: file1, box1, file2, box2
        result_df = filtered_df[['Scroll', 'Frg', 'File_x', 'Box_x', 'File_y', 'Box_y']]
        return result_df

    def get_image_path(self, file_name, box):
        """Generate the image path based on file and box information."""
        image_file = f"{file_name}_{int(box)}.jpg"
        #return os.path.join(self.image_base_path, file_name, image_file)
        return image_file

    def process_and_save_matches(self, output_filename, top_n=-1):
        """Process matches, save the updated CSV file in chunks, and sort the results."""
        matches_df = self.get_matched_pam_files()

        # Create a temporary file for the intermediate results
        temp_output = output_filename + '.temp'

        # Process and save matches as before
        with open(temp_output, 'w') as f:
            header_written = False
            for chunk in self.read_csv_chunks():
                chunk['Match'] = 0

                for index, row in tqdm(matches_df.iterrows(), total=matches_df.shape[0], desc="Processing Matches"):
                    image_path1 = self.get_image_path(row['File_x'], row['Box_x'])
                    image_path2 = self.get_image_path(row['File_y'], row['Box_y'])

                    # Check if both image paths are present in the current chunk
                    matching_rows = chunk[((chunk['file1'] == image_path1) & (chunk['file2'] == image_path2)) |
                                          ((chunk['file2'] == image_path1) & (chunk['file1'] == image_path2))].index
                    if len(matching_rows) >0:
                        chunk.loc[matching_rows, 'Match'] = 1
                        print(f"Match found: {image_path1} - {image_path2}")
                # Save the chunk to the output file
                if not header_written:
                    chunk.to_csv(f, index=False)
                    header_written = True
                else:
                    chunk.to_csv(f, index=False, header=False, mode='a')

        # Now read the temporary file, sort it, and save the top N results
        print("Processing final sorting...")
        df = pd.read_csv(temp_output)

        # Sort by the third column in descending order
        df_sorted = df.sort_values(by=df.columns[2], ascending=False)

        # Select the top N rows
        if (top_n > 0):
            df_sorted_top = df_sorted.head(top_n)
        else:
            df_sorted_top = df_sorted

        # Save the final sorted results
        df_sorted_top.to_csv(output_filename, index=False)

        # Clean up the temporary file
        os.remove(temp_output)

        print(f"Processed and sorted results saved to: {output_filename}")


if __name__ == "__main__":
    load_dotenv()

    # Access the environment variables
    base_path = os.getenv('BASE_PATH')
    PATCHES_IN = os.path.join(base_path, os.getenv('PATCHES_IN'))
    CSV_IN = os.path.join(base_path, os.getenv('CSV_IN'))
    _sift_matches = os.path.join(base_path,  os.getenv('SIFT_MATCHES'))
    _sift_matches_w_tp = os.getenv('SIFT_MATCHES_W_TP')

    # Define output filename
    output_filename = os.path.join(base_path, _sift_matches_w_tp)

    # Initialize and run the processor
    pam_processor = PamProcessor(PATCHES_IN, CSV_IN, _sift_matches)
    pam_processor.process_and_save_matches(output_filename, top_n=-1)