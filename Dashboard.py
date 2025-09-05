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

from streamlit_autorefresh import st_autorefresh
from tensorboard.backend.event_processing import event_accumulator

# Use a generic logger for simplicity. We don't have easy access to the main config file here.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL_CONFIG = {
    "TENSORBOARD_DATA": 300,
    "VIDEO_COMPILATION": 60,
}

def _get_dynamic_framerate(frame_count: int) -> float:
    """
    Calculates a dynamic framerate based on the total number of frames.

    This slows down videos with fewer frames (to show detail) and speeds
    up longer videos (to keep the duration manageable).

    Args:
        frame_count: The total number of frames.

    Returns:
        The suggested framerate in frames per second.
    """
    # Configuration of (max_frames, framerate) pairs, sorted by max_frames.
    FRAMERATE_THRESHOLDS = [
        (50, 0.5),
        (200, 1.0),
        (1000, 10.0),
        (2400, 24.0)
    ]
    DEFAULT_FRAMERATE = 24.0

    for max_frames, rate in FRAMERATE_THRESHOLDS:
            if frame_count <= max_frames:
                return rate
    return DEFAULT_FRAMERATE

@st.cache_data(ttl=CACHE_TTL_CONFIG["VIDEO_COMPILATION"])
def _compile_video(frames_path, video_output_path, frame_count_mtime_tuple):
    """
    Compiles PNG frames into an MP4 video using FFmpeg.
    This function is cached and only reruns if the number or last-modified time of frames changes.
    """
    frame_count = frame_count_mtime_tuple[0]
    if not os.path.isdir(frames_path) or frame_count == 0:
        return None, "No frames found to compile."

    framerate = _get_dynamic_framerate(frame_count)
    logger.info(f"Compiling {frame_count} frames into video at {framerate} fps...")
    input_pattern = os.path.join(frames_path, '*.png')
    
    command = [
        'ffmpeg',
        '-y',                         # Overwrite output file if it exists
        '-framerate', str(framerate), # Set the INPUT framerate
        '-pattern_type', 'glob',      # Use glob pattern matching for input
        '-i', input_pattern,          # Input files (e.g., 'path/*.png')
        '-r', str(framerate),         # Set the OUTPUT video framerate to match
        '-c:v', 'libx264',            # Video codec
        '-pix_fmt', 'yuv420p',        # Pixel format for compatibility
        video_output_path
    ]
    try:
        # We use DEVNULL to hide the verbose ffmpeg output from the console.
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
    @st.cache_data(ttl=CACHE_TTL_CONFIG["TENSORBOARD_DATA"])
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
        cluster_text = None
    
        for event_file in all_event_files:
            try:
                size_guidance = {
                    "scalars": 0,  # Load all scalars.
                    "tensors": 0,  # Load all tensors, for text summaries.
                }
                ea = event_accumulator.EventAccumulator(event_file, size_guidance=size_guidance)
                ea.Reload()

                scalar_tags = ea.Tags().get("scalars", [])
                if scalar_tags:
                    parent_dir_path = os.path.dirname(event_file)
                    parent_dir_name = os.path.basename(parent_dir_path)
                    category_tag = scalar_tags[0]
                    metric_name = parent_dir_name.replace(f"{category_tag}_", "", 1) if parent_dir_name != 'tensorboard_logs' else category_tag
                    events = ea.Scalars(category_tag)
                    if events:
                        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
                        df['metric'] = metric_name
                        scalar_data[category_tag].append(df)

                tensor_tags = ea.Tags().get('tensors', [])
                cluster_analysis_tag = next((tag for tag in tensor_tags if tag.startswith('Cluster_Analysis')), None)

                if cluster_analysis_tag:
                    text_events = ea.Text(cluster_analysis_tag)
                    if text_events:
                        cluster_text = text_events[-1].text_data.decode('utf-8')
        
            except Exception as e:
                logger.warning(f"Skipping corrupted or incomplete event file {os.path.basename(event_file)}: {e}")
                pass

        dataframes = {
            category: pd.concat(dfs, ignore_index=True)
            for category, dfs in scalar_data.items() if dfs
        }

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
            x=alt.X('step:Q', title='Global Step'),
            y=alt.Y('value:Q', title='Value', scale=alt.Scale(zero=False)),
            color=alt.Color('metric:N', title='Metric'),
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
            loss_key = tags.get("LOSSES", "Losses")
            lr_key = tags.get("LEARNING_RATE", "Learning Rate")
            physics_key = tags.get("PHYSICS_PARAMS", "Physics Parameters")
            hormone_key = tags.get("HORMONE_LEVELS", "Hormone Levels")

            log_dir = os.path.join(selected_experiment_path, 'tensorboard_logs')
            all_event_files = glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True)
            file_stats = tuple((os.path.getmtime(f), os.path.getsize(f)) for f in all_event_files) if all_event_files else None
            
            with st.spinner("Checking for new TensorBoard data..."):
                all_data, cluster_text = self._load_tensorboard_data(selected_experiment_path, file_stats)

            if not all_data:
                st.warning("No TensorBoard data found for this experiment yet. Waiting for logs...")
                st.stop()

            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Key Metrics", 
                "🌌 Semantic Space", 
                "🧬 Endocrine System", 
                "⚡ Embedding Projector"
            ])

            with tab1:
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
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Evolution of 2D Semantic Space")
                    frames_path = os.path.join(selected_experiment_path, "semantic_space_frames")
                    video_output_path = os.path.join(frames_path, "evolution.mp4")
                    frame_files = sorted(glob.glob(os.path.join(frames_path, "frame_*.png")))
                    frame_count = len(frame_files)
                    latest_mtime = os.path.getmtime(frame_files[-1]) if frame_files else 0
                    frame_count_mtime_tuple = (frame_count, latest_mtime)
                    compiled_path, error = _compile_video(frames_path, video_output_path, frame_count_mtime_tuple)

                    if compiled_path:
                        st.video(compiled_path)
                        st.caption(f"Live video of the 2D projection, compiled from {frame_count} frames.")
                    elif error:
                        st.error(error)
                    else:
                        st.info("No semantic space visualizations found yet. Waiting for frames...")

                    st.subheader("Live 8D Cluster Centroid Analysis")
                    if cluster_text:
                        st.markdown(cluster_text)
                    else:
                        st.info("Cluster analysis text not yet available.")
                
                with col2:
                    st.subheader("What is the Semantic Space?")
                    st.markdown("""
                    The Semantic Space is an 8-dimensional vector space where each concept (word) is treated as a particle with a position and mass. The model learns to arrange these particles according to a defined theoretical framework, simulating forces of attraction and repulsion to form meaningful clusters. The video shows a 2D projection of this dynamic space.

                    The space is structured into four theoretical planes:
                    """)
                    st.markdown("""
                    #### I. Core Conceptual Plane
                    *Defines the intrinsic nature of a concept.*
                    - **G-Axis (Groundedness):** Measures the tie to sensory-motor experience. 
                        - *Low G:* Concrete & Sensory (rock, water)
                        - *High G:* Abstract & Relational (justice, freedom)
                    - **X-Axis (Taxonomic Specificity):** Measures the concept's level in a hierarchy.
                        - *Low X:* General (animal, tool)
                        - *High X:* Specific (poodle, screwdriver)

                    #### II. Affective Plane (Emotion)
                    *Maps core emotional qualities based on the Circumplex Model.*
                    - **V-Axis (Valence):** Measures positive vs. negative emotional charge.
                        - *Low V:* Negative (pain, sad)
                        - *High V:* Positive (joy, peace)
                    - **A-Axis (Arousal):** Measures emotional intensity.
                        - *Low A:* Calm (sleep, serene)
                        - *High A:* Excited (panic, rage)

                    #### III. Interoceptive Plane (Bodily State)
                    *Maps underlying physiological states.*
                    - **H-Axis (Homeostasis):** Measures internal balance vs. dysregulation.
                        - *Low H:* Dysregulated (nausea, stress)
                        - *High H:* Regulated (healthy, energized)
                    - **S-Axis (Somatic Focus):** Measures if an experience is more in the body or mind.
                        - *Low S:* Cognitive (belief, memory)
                        - *High S:* Somatic/Physical (pain, warmth)

                    #### IV. Structural Plane (Contextual Role)
                    *Defines a concept's role within a larger structure.*
                    - **F-Axis (Functional Role):** Measures purpose as meaning vs. grammar.
                        - *Low F:* Lexical/Content (mountain, sun)
                        - *High F:* Functional/Grammar (is, the)
                    - **T-Axis (Temporal Orientation):** Measures inherent relationship to time.
                        - *Low T:* Past-Oriented (history, memory)
                        - *High T:* Future-Oriented (plan, hope)
                    """)

            with tab3:
                st.subheader("Semantic Endocrine System")
                if hormone_key in all_data:
                    st.altair_chart(self._plot_chart(all_data[hormone_key], 'Hormone Levels Over Time'), use_container_width=True)
                else:
                    st.info(f"Hormone level data ('{hormone_key}') not available.")

            with tab4:
                st.subheader("🚀 Explore 8D Embeddings with TensorBoard Projector")
                st.markdown("""
                The **Embedding Projector** is a powerful tool within TensorBoard that lets you visualize and interact with the high-dimensional (8D) semantic space.
                - **Visualize**: See how concepts cluster together in 3D space.
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

if __name__ == '__main__':
    dashboard = TrainingDashboard()
    dashboard.run()