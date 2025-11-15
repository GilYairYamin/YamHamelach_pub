import ast
import json
import os
import pickle
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from functools import lru_cache
from tqdm import tqdm


class SIFTMatchExplorer:
    def __init__(self, debug_mode: bool = False):
        """Initialize the SIFT Match Explorer application."""
        print("[DEBUG] Initializing SIFT Match Explorer...")
        
        # Get script directory
        script_dir = Path(__file__).parent
        
        # Define possible .env file locations
        possible_env_locations = [
            script_dir / '.env',
            Path.cwd() / '.env',
            Path.home() / '.env'
        ]
        
        print("\n[DEBUG] Checking for .env files in:")
        for loc in possible_env_locations:
            print(f"- {loc}: {'EXISTS' if loc.exists() else 'NOT FOUND'}")
        
        # Load .env file from the script's directory
        env_path = script_dir / '.env'
        print("\n[DEBUG] Loading .env from:", env_path)
        
        # Print the contents of the .env file if it exists
        if env_path.exists():
            print("\n[DEBUG] Contents of .env file:")
            with open(env_path, 'r') as f:
                print(f.read())
        else:
            print("\n[DEBUG] WARNING: .env file not found at:", env_path)
        
        # Load environment variables
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Debug print all relevant environment variables
        print("\n[DEBUG] Environment variables after loading .env:")
        for key in ["BASE_PATH", "IMAGES_IN", "PATCHES_IN", "SIFT_MATCHES_CSV", "PATCHES_CACHE"]:
            print(f"{key}: {os.getenv(key)}")
        
        # Set up paths
        self.base_path = Path(os.getenv("BASE_PATH", "."))
        self.images_path = self.base_path / os.getenv("IMAGES_IN", "original_images")
        self.patches_path = self.base_path / os.getenv("PATCHES_IN", "patches")
        self.cache_path = self.base_path / os.getenv("PATCHES_CACHE", "patches_keypoints_cache")
        self.csv_path = self.base_path / "sift_matches_v3_w_tp_w_homo_3COL.csv"
        
        print(f"[DEBUG] Base path: {self.base_path}")
        print(f"[DEBUG] Images path: {self.images_path}")
        print(f"[DEBUG] Patches path: {self.patches_path}") 
        print(f"[DEBUG] Cache path: {self.cache_path}")
        print(f"[DEBUG] CSV path: {self.csv_path}")
        
        # Configuration
        self.debug = os.getenv("DEBUG", "False").lower() in ["true", "1", "t"]
        self.debug_mode = debug_mode  # New debug mode flag
        self.colors = ["yellow", "red", "blue", "green", "orange", "purple", "cyan", "magenta"]
        
        # Load and prepare data
        print("[DEBUG] Loading matches data...")
        self.matches_df = self._load_matches()
        print(f"[DEBUG] Loaded {len(self.matches_df)} matches from CSV")
        
        print("[DEBUG] Applying advanced filtering...")
        self.filtered_df = self._apply_advanced_filtering(self.matches_df)
        print(f"[DEBUG] Filtered to {len(self.filtered_df)} matches")
        
        print("[DEBUG] Preparing image matches...")
        self.image_matches = self._prepare_image_matches()
        self.available_images = sorted(list(self.image_matches.keys()))
        print(f"[DEBUG] Found {len(self.available_images)} available images")
        
        # Cache for loaded data
        self.cache = {}
        
        # Debug mode
        self.debug_info = []
        print("[DEBUG] Initialization complete")
    
    def _max_acceptable_error(self, poi_matches: int) -> float:
        """Calculate maximum acceptable error based on number of matches."""
        if poi_matches < 10:
            return 0
        return (1 - math.pow(math.e, -0.025 * poi_matches)) * 100
    
    def _apply_advanced_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced filtering based on error thresholds."""
        # Add column for dynamic error threshold
        df['max_error_threshold'] = df.get('distance', df.get('len_homo_err', 0)).apply(self._max_acceptable_error)
        
        # Filter based on mean homography error if available
        if 'mean_homo_err' in df.columns:
            filtered_df = df[df['mean_homo_err'] <= df['max_error_threshold']]
        else:
            filtered_df = df
        
        return filtered_df
    
    @lru_cache(maxsize=100)
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image and convert to RGB."""
        img = cv2.imread(str(image_path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    
    @lru_cache(maxsize=200)
    def _load_keypoints(self, pkl_file: str) -> List:
        """Load keypoints from pickle file."""
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        return data["keypoints"]
    
    @lru_cache(maxsize=200)
    def _get_patch_info(self, image_id: str, patch_id: str) -> Optional[Dict]:
        """Get patch information from JSON file."""
        json_file = self.base_path / image_id / f"{image_id}_patch_info.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                patch_info = json.load(f)
            return patch_info.get(patch_id)
        return None
    
    def _load_matches(self) -> pd.DataFrame:
        """Load matches from CSV file."""
        print(f"[DEBUG] Loading matches from {self.csv_path}")
        print("[DEBUG] Reading CSV file...")
        
        if self.debug_mode:
            print("[DEBUG] Debug mode enabled - loading only first 100 images")
            # Read the CSV file in chunks to find the first 100 unique images
            unique_images = set()
            chunk_size = 10000  # Process 10,000 rows at a time
            chunks = []
            
            for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
                # Get unique image IDs from this chunk
                chunk_images = set()
                for file1, file2 in zip(chunk['file1'], chunk['file2']):
                    img1_id = file1.split('_')[0]
                    img2_id = file2.split('_')[0]
                    chunk_images.add(img1_id)
                    chunk_images.add(img2_id)
                
                # Add new unique images to our set
                unique_images.update(chunk_images)
                
                # If we have enough images, stop reading
                if len(unique_images) >= 100:
                    break
            
            # Take only the first 100 images
            unique_images = list(unique_images)[:100]
            print(f"[DEBUG] Found {len(unique_images)} unique images")
            
            # Read the full CSV file but only keep rows with our selected images
            df = pd.read_csv(self.csv_path)
            df = df[df['file1'].str.split('_').str[0].isin(unique_images) & 
                   df['file2'].str.split('_').str[0].isin(unique_images)]
        else:
            df = pd.read_csv(self.csv_path)
        
        print(f"[DEBUG] Loaded {len(df)} matches from CSV")
        
        # Parse matches column if it exists and is string
        if 'matches' in df.columns and df['matches'].dtype == 'object':
            print("[DEBUG] Parsing matches column...")
            df['matches'] = df['matches'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            print("[DEBUG] Finished parsing matches column")
        
        # Sort by appropriate metric
        if 'mean_homo_err' in df.columns:
            print("[DEBUG] Sorting by mean_homo_err...")
            df = df.sort_values(by='mean_homo_err', ascending=True)
        elif 'distance' in df.columns:
            print("[DEBUG] Sorting by distance...")
            df = df.sort_values(by='distance', ascending=True)
        
        return df
    
    def _prepare_image_matches(self) -> Dict[str, List[Dict]]:
        """Prepare matches grouped by image pairs."""
        print("[DEBUG] Preparing image matches...")
        image_matches = {}
        image_distances = {}  # New dictionary to store minimum distances between images
        
        # Create progress bar for processing matches
        total_matches = len(self.filtered_df)
        print(f"[DEBUG] Processing {total_matches} matches...")
        
        for idx, row in tqdm(self.filtered_df.iterrows(), total=total_matches, desc="Processing matches"):
            # Extract image IDs from filenames
            img1_id = row['file1'].split('_')[0]
            img2_id = row['file2'].split('_')[0]
            
            # Calculate distance score
            score = float('inf')
            if 'mean_homo_err' in row:
                score = row['mean_homo_err']
            elif 'distance' in row:
                score = row['distance']
            
            # Update minimum distance between image pairs
            pair_key = tuple(sorted([img1_id, img2_id]))
            if pair_key not in image_distances or score < image_distances[pair_key]:
                image_distances[pair_key] = score
            
            # Create keys for both directions
            for img_id in [img1_id, img2_id]:
                if img_id not in image_matches:
                    image_matches[img_id] = []
                
                match_info = {
                    'file1': row['file1'],
                    'file2': row['file2'],
                    'img1_id': img1_id,
                    'img2_id': img2_id
                }
                
                # Add all available metrics
                for col in ['distance', 'mean_homo_err', 'sum_homo_err', 'len_homo_err', 'std_homo_err']:
                    if col in row:
                        match_info[col] = row[col]
                
                if 'matches' in row:
                    match_info['matches'] = row['matches']
                
                image_matches[img_id].append(match_info)
        
        # Store the distances in the class for later use
        self.image_distances = image_distances
        print(f"[DEBUG] Prepared matches for {len(image_matches)} images")
        return image_matches
    
    def get_related_images(self, img_id: str, max_distance: float = 50.0) -> List[str]:
        """Get list of images that have matches with distance <= max_distance."""
        related_images = []
        for (img1, img2), distance in self.image_distances.items():
            if img1 == img_id and distance <= max_distance:
                related_images.append(img2)
            elif img2 == img_id and distance <= max_distance:
                related_images.append(img1)
        return sorted(list(set(related_images)))
    
    def _get_image_filename(self, image_id: str) -> str:
        """Get the full image filename from ID."""
        return f"{image_id}.jpg"
    
    def _normalize_coordinates(self, coords: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
        """Normalize coordinates for proper display."""
        return (coords[0] / width, coords[1] / height)
    
    def plot_image_with_patches(self, image_id: str, matches: List[Dict] = None, show_scores: bool = True) -> go.Figure:
        """Plot an image with patch rectangles and match information."""
        img_path = self.images_path / self._get_image_filename(image_id)
        img = self._load_image(str(img_path))
        
        if img is None:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add the image
        fig.add_trace(go.Image(z=img))
        
        # Get image dimensions
        height, width = img.shape[:2]
        is_vertical = height > width
        
        # Calculate appropriate figure size
        if is_vertical:
            fig.update_layout(width=600, height=800)
        else:
            fig.update_layout(width=800, height=600)
        
        # Get all patches for this image with their matches
        if matches:
            # Group matches by patch for color coding
            patch_matches = {}
            for i, match in enumerate(matches):
                if match['img1_id'] == image_id:
                    patch_file = match['file1']
                else:
                    patch_file = match['file2']
                
                if patch_file not in patch_matches:
                    patch_matches[patch_file] = []
                patch_matches[patch_file].append((i, match))
            
            # Draw rectangles for matched patches
            color_idx = 0
            for patch_file, match_list in patch_matches.items():
                patch_id = patch_file.split('_')[1].split('.')[0]
                patch_info = self._get_patch_info(image_id, patch_id)
                
                if patch_info:
                    coords = patch_info['coordinates']
                    color = self.colors[color_idx % len(self.colors)]
                    
                    # Draw rectangle
                    fig.add_shape(
                        type="rect",
                        x0=coords[0], y0=coords[1],
                        x1=coords[2], y1=coords[3],
                        line=dict(color=color, width=3),
                        fillcolor="rgba(0,0,0,0)"
                    )
                    
                    # Add patch number
                    center_x = (coords[0] + coords[2]) / 2
                    center_y = (coords[1] + coords[3]) / 2
                    
                    fig.add_annotation(
                        x=center_x, y=center_y,
                        text=patch_id,
                        showarrow=False,
                        font=dict(size=12, color="white"),
                        bgcolor=color,
                        opacity=0.8
                    )
                    
                    # Add score information if requested
                    if show_scores and match_list:
                        scores = []
                        for _, match in match_list[:3]:  # Show top 3 scores
                            if 'mean_homo_err' in match:
                                scores.append(f"{match['mean_homo_err']:.2f}")
                            elif 'distance' in match:
                                scores.append(f"{match['distance']:.2f}")
                        
                        if scores:
                            score_text = "Scores: " + ", ".join(scores)
                            fig.add_annotation(
                                x=center_x, y=coords[3] + 10,
                                text=score_text,
                                showarrow=False,
                                font=dict(size=10, color="black"),
                                bgcolor="white",
                                opacity=0.8
                            )
                    
                    color_idx += 1
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x")
        )
        
        return fig
    
    def plot_patch_matches(self, match: Dict, max_keypoints: int = 100) -> go.Figure:
        """Plot two patches with their matched keypoints and connecting lines."""
        file1, file2 = match['file1'], match['file2']
        
        # Load patches
        patch1_path = self.patches_path / file1.split('_')[0] / file1
        patch2_path = self.patches_path / file2.split('_')[0] / file2
        
        patch1 = self._load_image(str(patch1_path))
        patch2 = self._load_image(str(patch2_path))
        
        if patch1 is None or patch2 is None:
            return go.Figure()
        
        # Create combined image for side-by-side display
        combined_width = patch1.shape[1] + patch2.shape[1] + 20  # 20px gap
        combined_height = max(patch1.shape[0], patch2.shape[0])
        combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Place patches
        combined_img[:patch1.shape[0], :patch1.shape[1]] = patch1
        combined_img[:patch2.shape[0], patch1.shape[1] + 20:] = patch2
        
        fig = go.Figure()
        fig.add_trace(go.Image(z=combined_img))
        
        # Load keypoints if available
        if 'matches' in match and match['matches']:
            kp1_path = self.cache_path / f"{file1}.pkl"
            kp2_path = self.cache_path / f"{file2}.pkl"
            
            try:
                keypoints1 = self._load_keypoints(str(kp1_path))
                keypoints2 = self._load_keypoints(str(kp2_path))
                matches_list = match['matches']
                
                # Plot keypoints and connections
                for i, (idx1, idx2, _) in enumerate(matches_list[:max_keypoints]):
                    kp1 = keypoints1[idx1][0]
                    kp2 = keypoints2[idx2][0]
                    
                    # Adjust x coordinate for second patch
                    kp2_adjusted_x = kp2[0] + patch1.shape[1] + 20
                    
                    # Add keypoints
                    fig.add_trace(go.Scatter(
                        x=[kp1[0], kp2_adjusted_x],
                        y=[kp1[1], kp2[1]],
                        mode='markers+lines',
                        marker=dict(color='red', size=8),
                        line=dict(color='green', width=1),
                        showlegend=False,
                        opacity=0.7
                    ))
            except Exception as e:
                if self.debug:
                    self.debug_info.append(f"Error loading keypoints: {e}")
        
        # Add distance/score annotation
        score_text = ""
        if 'mean_homo_err' in match:
            score_text = f"Mean Error: {match['mean_homo_err']:.2f}"
        elif 'distance' in match:
            score_text = f"Distance: {match['distance']:.2f}"
        
        if score_text:
            fig.add_annotation(
                x=combined_width/2, y=10,
                text=score_text,
                showarrow=False,
                font=dict(size=16, color="white"),
                bgcolor="black",
                opacity=0.8
            )
        
        # Add patch labels
        fig.add_annotation(
            x=patch1.shape[1]/2, y=patch1.shape[0] + 10,
            text=f"Patch: {file1}",
            showarrow=False,
            font=dict(size=12, color="black")
        )
        
        fig.add_annotation(
            x=patch1.shape[1] + 20 + patch2.shape[1]/2, y=patch2.shape[0] + 10,
            text=f"Patch: {file2}",
            showarrow=False,
            font=dict(size=12, color="black")
        )
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        
        return fig
    
    def get_matches_for_images(self, img1_id: str, img2_id: str) -> List[Dict]:
        """Get all matches between two images."""
        matches = []
        
        for match in self.image_matches.get(img1_id, []):
            if (match['img1_id'] == img2_id or match['img2_id'] == img2_id):
                matches.append(match)
        
        # Sort by best score
        if matches and 'mean_homo_err' in matches[0]:
            matches.sort(key=lambda x: x.get('mean_homo_err', float('inf')))
        elif matches and 'distance' in matches[0]:
            matches.sort(key=lambda x: x.get('distance', float('inf')))
        
        return matches
    
    def calculate_match_statistics(self, matches: List[Dict]) -> Dict[str, float]:
        """Calculate statistics for a set of matches."""
        if not matches:
            return {}
        
        stats = {}
        
        # Calculate average of top matches
        score_key = 'mean_homo_err' if 'mean_homo_err' in matches[0] else 'distance'
        scores = [m[score_key] for m in matches if score_key in m]
        
        if scores:
            top_scores = scores[:min(4, len(scores))]
            stats['average_top_4'] = sum(top_scores) / len(top_scores)
            stats['best_score'] = min(scores)
            stats['worst_score'] = max(scores)
            stats['total_matches'] = len(matches)
        
        return stats
    
    def update_images(self, img1_id: str, img2_id: str, error_threshold: float = None):
        """Update the image displays when selection changes."""
        print(f"[DEBUG] Updating images for {img1_id} and {img2_id} with threshold {error_threshold}")
        
        if not img1_id or not img2_id:
            print("[DEBUG] No images selected")
            return go.Figure(), go.Figure(), None, "No matches found", ""
        
        # Get matches between selected images
        matches = self.get_matches_for_images(img1_id, img2_id)
        print(f"[DEBUG] Found {len(matches)} initial matches")
        
        # Apply error threshold filter if specified
        if error_threshold is not None and matches:
            score_key = 'mean_homo_err' if 'mean_homo_err' in matches[0] else 'distance'
            matches = [m for m in matches if m.get(score_key, float('inf')) <= error_threshold]
            print(f"[DEBUG] After threshold filtering: {len(matches)} matches")
        
        if not matches:
            print("[DEBUG] No matches after filtering")
            fig1 = self.plot_image_with_patches(img1_id)
            fig2 = self.plot_image_with_patches(img2_id)
            return fig1, fig2, None, "No matches found between these images", ""
        
        # Calculate statistics
        stats = self.calculate_match_statistics(matches)
        print(f"[DEBUG] Calculated statistics: {stats}")
        
        # Create visualizations
        print("[DEBUG] Creating visualizations...")
        fig1 = self.plot_image_with_patches(img1_id, matches)
        fig2 = self.plot_image_with_patches(img2_id, matches)
        
        # Create dropdown choices for matches
        match_choices = []
        for i, match in enumerate(matches):
            score_key = 'mean_homo_err' if 'mean_homo_err' in match else 'distance'
            score = match.get(score_key, 0)
            match_choices.append(
                (f"Match {i+1}: {match['file1']} ↔ {match['file2']} (score: {score:.2f})", i)
            )
        
        # Create statistics text
        stats_text = f"""
        Total Matches: {stats.get('total_matches', 0)}
        Best Score: {stats.get('best_score', 0):.2f}
        Average Top 4: {stats.get('average_top_4', 0):.2f}
        """
        
        print("[DEBUG] Update complete")
        return (
            fig1, 
            fig2, 
            gr.Dropdown.update(choices=match_choices, value=0, visible=True),
            f"Found {len(matches)} matches",
            stats_text
        )
    
    def show_match_details(self, img1_id: str, img2_id: str, match_idx: int, error_threshold: float = None):
        """Show detailed view of a selected match."""
        print(f"[DEBUG] Showing match details for {img1_id} and {img2_id}, match index {match_idx}")
        
        if match_idx is None:
            print("[DEBUG] No match index provided")
            return go.Figure()
        
        matches = self.get_matches_for_images(img1_id, img2_id)
        print(f"[DEBUG] Found {len(matches)} matches")
        
        # Apply threshold filter
        if error_threshold is not None and matches:
            score_key = 'mean_homo_err' if 'mean_homo_err' in matches[0] else 'distance'
            matches = [m for m in matches if m.get(score_key, float('inf')) <= error_threshold]
            print(f"[DEBUG] After threshold filtering: {len(matches)} matches")
        
        if match_idx >= len(matches):
            print(f"[DEBUG] Match index {match_idx} out of range")
            return go.Figure()
        
        match = matches[match_idx]
        print(f"[DEBUG] Plotting patch matches for {match['file1']} and {match['file2']}")
        return self.plot_patch_matches(match)


def create_app(debug_mode: bool = False):
    """Create and launch the Gradio application."""
    explorer = SIFTMatchExplorer(debug_mode=debug_mode)
    
    with gr.Blocks(title="SIFT Match Explorer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SIFT Match Explorer")
        gr.Markdown("Advanced visualization tool for exploring SIFT matches between image patches.")
        
        with gr.Row():
            with gr.Column(scale=1):
                img1_dropdown = gr.Dropdown(
                    choices=explorer.available_images,
                    label="Select Image A",
                    value=explorer.available_images[0] if explorer.available_images else None
                )
            with gr.Column(scale=1):
                img2_dropdown = gr.Dropdown(
                    choices=explorer.available_images,
                    label="Select Image B",
                    value=explorer.available_images[1] if len(explorer.available_images) > 1 else None
                )
            with gr.Column(scale=1):
                error_threshold = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,  # Changed default to 50
                    step=1,
                    label="Error Threshold",
                    info="Filter matches by maximum error"
                )
        
        with gr.Row():
            img1_display = gr.Plot(label="Image A with Patches")
            img2_display = gr.Plot(label="Image B with Patches")
        
        with gr.Row():
            with gr.Column(scale=3):
                match_dropdown = gr.Dropdown(
                    label="Select Match to View Details",
                    visible=False
                )
            with gr.Column(scale=1):
                stats_display = gr.Textbox(
                    label="Match Statistics",
                    interactive=False,
                    lines=3
                )
        
        status_text = gr.Textbox(label="Status", interactive=False)
        patch_display = gr.Plot(label="Detailed Patch Match View")
        
        # Debug information (only shown in debug mode)
        if explorer.debug:
            debug_display = gr.Textbox(
                label="Debug Information",
                interactive=False,
                lines=10,
                visible=True
            )
        
        # Event handlers
        def on_image1_change(img1, threshold):
            """Update available choices in image2 dropdown based on image1 selection."""
            if not img1:
                return gr.Dropdown.update(choices=explorer.available_images), gr.Plot(), gr.Plot(), None, "No image selected", ""
            
            # Get related images with distance <= threshold
            related_images = explorer.get_related_images(img1, threshold)
            print(f"[DEBUG] Found {len(related_images)} related images for {img1} with threshold {threshold}")
            
            # Update image2 dropdown choices
            img2_choices = related_images if related_images else explorer.available_images
            img2_value = img2_choices[0] if img2_choices else None
            
            # Update displays
            fig1, fig2, match_drop, status, stats = explorer.update_images(img1, img2_value, threshold)
            
            return (
                gr.Dropdown.update(choices=img2_choices, value=img2_value),
                fig1, fig2, match_drop, status, stats
            )
        
        def on_image2_change(img1, img2, threshold):
            """Update displays when image2 is changed."""
            return explorer.update_images(img1, img2, threshold)
        
        def on_match_select(img1, img2, match_idx, threshold):
            return explorer.show_match_details(img1, img2, match_idx, threshold)
        
        # Wire up events
        img1_dropdown.change(
            fn=on_image1_change,
            inputs=[img1_dropdown, error_threshold],
            outputs=[img2_dropdown, img1_display, img2_display, match_dropdown, status_text, stats_display]
        )
        
        img2_dropdown.change(
            fn=on_image2_change,
            inputs=[img1_dropdown, img2_dropdown, error_threshold],
            outputs=[img1_display, img2_display, match_dropdown, status_text, stats_display]
        )
        
        error_threshold.change(
            fn=on_image1_change,
            inputs=[img1_dropdown, error_threshold],
            outputs=[img2_dropdown, img1_display, img2_display, match_dropdown, status_text, stats_display]
        )
        
        match_dropdown.change(
            fn=on_match_select,
            inputs=[img1_dropdown, img2_dropdown, match_dropdown, error_threshold],
            outputs=[patch_display]
        )
        
        # Initial load
        app.load(
            fn=on_image1_change,
            inputs=[img1_dropdown, error_threshold],
            outputs=[img2_dropdown, img1_display, img2_display, match_dropdown, status_text, stats_display]
        )
    
    return app


if __name__ == "__main__":
    app = create_app(debug_mode=True)  # Enable debug mode by default
    app.launch(share=False, debug=True)