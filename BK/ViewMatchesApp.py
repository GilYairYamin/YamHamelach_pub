import os

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

# Load the CSV file
df = pd.read_csv("matches.csv")

# Add a match_id column if not already present
df.reset_index(inplace=True)
df.rename(columns={"index": "match_id"}, inplace=True)
df["match_id"] = df["match_id"] + 1  # To start match_id from 1 instead of 0

st.title("Image Matches Viewer")

# Display Matches Data and the match selection side-by-side
col1, col2 = st.columns(2)

# Column 1: Display the DataFrame
with col1:
    st.write("### Select a Match to View")
    selected_scroll = st.selectbox("Select Scroll", df["scroll"].unique())

    filtered_df = df[df["scroll"] == selected_scroll]

    selected_fragment = st.selectbox(
        "Select Fragment", filtered_df["fragment"].unique()
    )

    filtered_df = filtered_df[filtered_df["fragment"] == selected_fragment]

    st.write("### Matches Data")
    st.dataframe(df)

# Column 2: Allow user to select a match based on 'scroll' and 'fragment'
with col2:
    if not filtered_df.empty:
        match_options = filtered_df["match_id"].tolist()
        match_option_labels = [
            f"Match {row.match_id}: File1 {os.path.basename(row.file1)}, File2 {os.path.basename(row.file2)}, Distance {row.distance}"
            for _, row in filtered_df.iterrows()
        ]

        match_selection = st.selectbox(
            "Select a Match",
            options=match_options,
            format_func=lambda x: match_option_labels[match_options.index(x)],
        )

        # Get the selected match
        selected_match = df[df["match_id"] == match_selection].iloc[0]

        st.write("### Selected Match Details")
        st.write(selected_match)

        # Display the corresponding image
        image_path = os.path.join(
            "../match_visualizations", f"match_{match_selection}.png"
        )

        if os.path.exists(image_path):
            st.write("### Match Visualization")
            st.image(image_path, caption=f"Match {match_selection}")
        else:
            st.error("Image not found.")
    else:
        st.warning("No matches found for the selected scroll and fragment.")
