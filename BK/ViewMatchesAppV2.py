import os

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

# Path to your PAM CSV file
pam_csv_file = "/Users/assafspanier/Dropbox/MY_DOC/Teaching/JCE/Research/research2024.jce.ac.il/YamHamelach_data_n_model/PAM_Identification_for_Asaf_Round_1_avinoamEdit.csv"

# Load the original PAM CSV file
pam_df = pd.read_csv(pam_csv_file)

st.title("PAM Identification Viewer")

# Display the original CSV using AgGrid
st.write("### PAM Data")

# Use AgGrid to display the DataFrame with selection enabled
gb = GridOptionsBuilder.from_dataframe(pam_df)
gb.configure_selection(selection_mode="single", use_checkbox=True)
grid_options = gb.build()

grid_response = AgGrid(
    pam_df,
    gridOptions=grid_options,
    height=400,
    width="100%",
    update_mode=GridUpdateMode.SELECTION_CHANGED,
)

selected_rows = grid_response["selected_rows"]

if selected_rows:
    selected_row = selected_rows[0]
    st.write("### Selected Row")
    st.write(selected_row)

    # Get the selected File and Box
    selected_File = str(selected_row["File"])
    selected_Box = str(selected_row["Box"])

    # Read the matches.csv
    matches_df = pd.read_csv("matches.csv")

    # Process matches_df to extract File1, Box1, File2, Box2
    def extract_file_box(file_path):
        filename = os.path.basename(file_path)
        # Assuming filename is 'FileName_Box.jpg'
        parts = filename.split("_")
        if len(parts) >= 2:
            file_name = parts[0]
            box_part = parts[1].split(".")[0]  # Remove .jpg extension
            box = box_part
            return file_name, box
        else:
            return None, None

    matches_df[["File1", "Box1"]] = matches_df.apply(
        lambda row: pd.Series(extract_file_box(row["file1"])), axis=1
    )
    matches_df[["File2", "Box2"]] = matches_df.apply(
        lambda row: pd.Series(extract_file_box(row["file2"])), axis=1
    )

    # Filter matches where selected File and Box match
    matches_filtered = matches_df[
        (
            (matches_df["File1"] == selected_File)
            & (matches_df["Box1"] == selected_Box)
        )
        | (
            (matches_df["File2"] == selected_File)
            & (matches_df["Box2"] == selected_Box)
        )
    ]

    if not matches_filtered.empty:
        st.write("### Matches Involving Selected File and Box")
        # Show the filtered matches
        # Use AgGrid to display and select a match
        gb_matches = GridOptionsBuilder.from_dataframe(matches_filtered)
        gb_matches.configure_selection(
            selection_mode="single", use_checkbox=True
        )
        grid_options_matches = gb_matches.build()

        grid_response_matches = AgGrid(
            matches_filtered,
            gridOptions=grid_options_matches,
            height=300,
            width="100%",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )

        selected_matches = grid_response_matches["selected_rows"]

        if selected_matches:
            selected_match = selected_matches[0]
            st.write("### Selected Match")
            st.write(selected_match)

            # Get the match_id
            match_id = selected_match["match_id"]

            # Load and display the corresponding image
            image_path = os.path.join(
                "../match_visualizations", f"match_{int(match_id)}.png"
            )
            if os.path.exists(image_path):
                st.image(image_path, caption=f"Match {match_id}")
            else:
                st.error(f"Image not found at {image_path}")
    else:
        st.warning("No matches found involving the selected File and Box.")
else:
    st.write("Select a row to view matches.")
