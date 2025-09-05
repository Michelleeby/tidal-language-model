import streamlit as st
import pandas as pd
import os
import glob
from ruamel.yaml import YAML
import subprocess
import altair as alt
import re
import collections
import subprocess
import logging

# Set up a generic logger for debugging the video compilation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from streamlit_autorefresh import st_autorefresh
from tensorboard.backend.event_processing import event_accumulator

@st.cache_data(ttl=60)
def _compile_video(frames_path, video_output_path, frame_count_mtime_tuple):
    """
    Compiles PNG frames into an MP4 video using FFmpeg.
    This function is cached and only reruns if the number or last-modified time of frames changes.
    """
    if not os.path.isdir(frames_path) or frame_count_mtime_tuple[0] == 0:
        return None, "No frames found to compile."

    logger.info(f"Compiling {frame_count_mtime_tuple[0]} frames into video...")
    
    # This command tells FFmpeg to:
    # -y: Overwrite output file if it exists
    # -framerate 10: Assume 10 frames per second for the input
    # -i ...: Read the PNG files in sequence
    # -c:v libx264: Use the efficient H.264 video codec
    # -pix_fmt yuv420p: Use a pixel format compatible with most web players
    command = [
        'ffmpeg',
        '-y',
        '-framerate', '30',
        '-i', os.path.join(frames_path, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_output_path
    ]
    try:
        # We use DEVNULL to hide the verbose ffmpeg output from the console
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Video compilation successful: {video_output_path}")
        return video_output_path, None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_msg = f"FFmpeg error: {e}. Is FFmpeg installed and in your PATH?"
        logger.error(error_msg)
        return None, error_msg

class TrainingDashboard:
    """
    A Streamlit-based dashboard to visualize the training process of the Tidal Language Model.
    It reads and parses TensorBoard logs to display metrics, parameters, and visualizations.
    """
    def __init__(self):
        st.set_page_config(
            page_title="Tidal LM Training Dashboard",
            page_icon="🌊",
            layout="wide"
        )

    @staticmethod
    @st.cache_data(ttl=300)
    def _load_tensorboard_data(experiment_path, file_stats):
        """
        Recursively finds and loads all TensorBoard event files, parsing directory
        names to correctly identify and separate individual metrics.
        """
        log_dir = os.path.join(experiment_path, 'tensorboard_logs')
        if not os.path.isdir(log_dir):
            return {}, None
    
        all_event_files = glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True)
        if not all_event_files:
            return {}, None
    
        scalar_data = collections.defaultdict(list)
    
        for event_file in all_event_files:
            try:
                # --- PARSE DIRECTORY FOR METRIC NAME ---
                parent_dir_path = os.path.dirname(event_file)
                parent_dir_name = os.path.basename(parent_dir_path)
    
                ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
                ea.Reload()
                
                if not ea.Tags()['scalars']:
                    continue
                
                # The tag inside the file is the general CATEGORY (e.g., "Physics")
                category_tag = ea.Tags()['scalars'][0]
    
                # Determine the specific METRIC name
                if parent_dir_name == 'tensorboard_logs':
                    # This is a top-level file (like Learning Rate)
                    # The category and metric are the same
                    metric_name = category_tag
                else:
                    # This is a subdirectory. The name is encoded in the directory.
                    # e.g., directory "Physics_G" and category "Physics" -> metric "G"
                    # We replace the category and the underscore to get the metric name.
                    metric_name = parent_dir_name.replace(f"{category_tag}_", "", 1)
    
    
                # --- DATA EXTRACTION ---
                events = ea.Scalars(category_tag)
                if not events:
                    continue
                
                df = pd.DataFrame(
                    [(event.step, event.value) for event in events],
                    columns=['step', 'value']
                )
                # Assign the CORRECT, specific metric name to the metric column
                df['metric'] = metric_name
                
                # Group the DataFrame by its general category
                scalar_data[category_tag].append(df)
    
            except Exception:
                # Silently ignore corrupted or empty files
                pass
            
        # --- AGGREGATION AND FINALIZATION ---
        dataframes = {
            category: pd.concat(dfs, ignore_index=True)
            for category, dfs in scalar_data.items()
        }
    
        # (Text data loading remains the same)
        cluster_text = None
        for event_file in all_event_files:
            try:
                ea = event_accumulator.EventAccumulator(event_file, size_guidance={'tensors': 0})
                ea.Reload()
                if 'Cluster_Analysis' in ea.Tags()['tensors']:
                    text_events = ea.Tensors('Cluster_Analysis')
                    if text_events:
                        cluster_text = text_events[-1].tensor_proto.string_val[0].decode('utf-8')
                        break
            except:
                continue
            
        return dataframes, cluster_text

    @staticmethod
    def _get_experiment_dirs():
        """Finds all experiment directories."""
        experiments_path = "experiments"
        if not os.path.isdir(experiments_path):
            return []
        dirs = [os.path.join(experiments_path, d) for d in os.listdir(experiments_path)]
        dirs = sorted([d for d in dirs if os.path.isdir(d)], key=os.path.getmtime, reverse=True)
        return [os.path.basename(d) for d in dirs]

    @staticmethod
    def _load_config(experiment_path):
        """Loads the YAML config for a given experiment."""
        config_path = os.path.join(experiment_path, 'config.yaml')
        if os.path.exists(config_path):
            yaml = YAML(typ='safe')
            with open(config_path, 'r') as f:
                return yaml.load(f)
        return None

    @staticmethod
    def _plot_chart(df, title):
        """Creates an Altair chart for a given DataFrame."""
        chart = alt.Chart(df).mark_line(interpolate='basis').encode(
            # The x (horizontal) axis is the training step.
            x=alt.X('step:Q', title='Global Step'),
            
            # The y (vertical) axis is the value of the metric.
            y=alt.Y('value:Q', title='Value', scale=alt.Scale(zero=False)),
            
            # This is the corrected line:
            # It tells Altair to create a separate colored line for each unique
            # entry in the 'metric' column (e.g., 'G', 'Temperature', 'Total Loss').
            color=alt.Color('metric:N', title='Metric'),

            # Tooltip for interactivity on hover.
            tooltip=['step', 'value', 'metric']
        ).properties(
            title=title
        ).interactive()
        return chart
    
    def run(self):
        """
        The main method to run the Streamlit application.
        """
        st.title("🌊 Tidal Language Model - Training Dashboard")
        st.markdown("Monitor and analyze the training loop of the Tidal Language Model.")
        
        # Set the refresh interval to 15 seconds. This will trigger a script rerun.
        # Thanks to caching, this is very efficient.
        refresh_interval_seconds = 15
        st_autorefresh(interval=refresh_interval_seconds * 1000, key="data_refresher")
        
        st.sidebar.header("Experiment Controls")
        experiment_dirs = self._get_experiment_dirs()

        if not experiment_dirs:
            st.error("No experiment directories found in the 'experiments' folder.")
            st.stop()

        selected_experiment_id = st.sidebar.selectbox("Select an Experiment ID", options=experiment_dirs, index=0)
        selected_experiment_path = os.path.join("experiments", selected_experiment_id)

        if st.sidebar.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.sidebar.markdown("---")
        config = self._load_config(selected_experiment_path)
        if config:
            st.sidebar.subheader("Experiment Config")
            st.sidebar.json(config, expanded=False)

        if selected_experiment_id:
            st.header(f"📊 Analysis for Experiment: `{selected_experiment_id}`")

            tags = (config.get("TENSORBOARD_TAGS", {}) if config else {})
            # These keys now map to the *category* part of the tag, e.g., "Loss" from "Loss/Total"
            loss_key = tags.get("LOSSES", "Losses")
            lr_key = tags.get("LEARNING_RATE", "Learning Rate")
            physics_key = tags.get("PHYSICS_PARAMS", "Physics Parameters")
            hormone_key = tags.get("HORMONE_LEVELS", "Hormone Levels")

            # We create a file_stats tuple from all found event files
            # to ensure the cache invalidates if any of them change.
            log_dir = os.path.join(selected_experiment_path, 'tensorboard_logs')
            all_event_files = glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True)
            file_stats = None
            if all_event_files:
                file_stats = tuple((os.path.getmtime(f), os.path.getsize(f)) for f in all_event_files)
            
            with st.spinner("Checking for new TensorBoard data..."):
                all_data, cluster_text = self._load_tensorboard_data(selected_experiment_path, file_stats)

            if not all_data:
                st.warning("No TensorBoard data found for this experiment yet. Waiting for logs...")
                st.stop()

            tab1, tab2, tab3, tab4 = st.tabs(["📈 Key Metrics", "🌌 2D Space Evolution", "🧬 8D Cluster Analysis", "⚡ Embedding Projector"])

            with tab1:
                # This code only runs if tab1 is active. The data is already loaded and cached,
                # so rendering the chart is extremely fast.
                st.subheader("Training Performance")
                col1, col2 = st.columns(2)
                with col1:
                    if loss_key in all_data:
                        st.altair_chart(self._plot_chart(all_data[loss_key], 'Losses Over Time'), use_container_width=True)
                    else: st.info(f"Loss data ('{loss_key}') not available.")
                with col2:
                    if lr_key in all_data:
                        st.altair_chart(self._plot_chart(all_data[lr_key], 'Learning Rate Schedule'), use_container_width=True)
                    else: st.info(f"Learning Rate data ('{lr_key}') not available.")
                
                st.subheader("Learnable Physics Parameters")
                if physics_key in all_data:
                     st.altair_chart(self._plot_chart(all_data[physics_key], 'Physics Parameter Evolution'), use_container_width=True)
                else: st.info(f"Physics parameter data ('{physics_key}') not available.")

            with tab2:
                st.subheader("Evolution of 2D Semantic Space")
                frames_path = os.path.join(selected_experiment_path, "semantic_space_frames")
                video_output_path = os.path.join(frames_path, "evolution.mp4")
                # Get a list of all frame files
                frame_files = sorted(glob.glob(os.path.join(frames_path, "frame_*.png")))
                frame_count = len(frame_files)
                # Get the modification time of the newest frame, or 0 if none exist
                latest_mtime = os.path.getmtime(frame_files[-1]) if frame_files else 0
                # This tuple is our cache key. If it changes, the video is recompiled.
                frame_count_mtime_tuple = (frame_count, latest_mtime)
                # Call the cached compilation function
                compiled_path, error = _compile_video(frames_path, video_output_path, frame_count_mtime_tuple)

                if compiled_path:
                    st.video(compiled_path)
                    st.caption(f"Live-updated video compiled from {frame_count} frames. Last updated: {pd.to_datetime(latest_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
                elif error:
                    st.error(error)
                else:
                    st.info("No semantic space visualizations found yet. Waiting for frames...")

            with tab3:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Semantic Endocrine System")
                    if hormone_key in all_data:
                        st.altair_chart(self._plot_chart(all_data[hormone_key], 'Hormone Levels Over Time'), use_container_width=True)
                    else:
                        st.info(f"Hormone level data ('{hormone_key}') not available.")
                
                with col2:
                    st.subheader("8D Cluster Centroid Analysis")
                    if cluster_text:
                        st.markdown(cluster_text)
                    else:
                        st.info("Cluster analysis text not yet available.")

            with tab4:
                st.subheader("🚀 Explore 8D Embeddings with TensorBoard Projector")
                st.markdown("""
                The **Embedding Projector** is a powerful tool within TensorBoard that lets you visualize and interact with the high-dimensional (8D) semantic space. Since your `Trainer` is already logging the final embeddings, you can explore them here.
                - **Visualize**: See how concepts cluster together in 3D space (using PCA, T-SNE, etc.).
                - **Search**: Find the nearest neighbors to a given word to understand its learned relationships.
                Click the button below to launch the full TensorBoard interface. The "PROJECTOR" tab will be available there.
                """)
                
                logdir = selected_experiment_path
                port = 6006
                
                if 'tensorboard_process' not in st.session_state:
                    st.session_state.tensorboard_process = None

                if st.button("Launch Projector"):
                    if st.session_state.tensorboard_process is not None:
                        try: st.session_state.tensorboard_process.kill()
                        except: pass

                    command = f"tensorboard --logdir={logdir} --port={port}"
                    proc = subprocess.Popen(command, shell=True)
                    st.session_state.tensorboard_process = proc
                    st.success(f"TensorBoard is launching. [Click here to open Projector](http://localhost:{port}/#projector)")
                    st.info("TensorBoard is running in a background process. You can close this browser tab, and it will continue to run in your terminal.")

if __name__ == '__main__':
    dashboard = TrainingDashboard()
    dashboard.run()