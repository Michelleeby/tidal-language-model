import streamlit as st
import pandas as pd
import os
import glob
from ruamel.yaml import YAML
import subprocess
import altair as alt
from tensorboard.backend.event_processing import event_accumulator
import re

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
    @st.cache_data(ttl=300) # Cache data for 5 minutes
    def _load_tensorboard_data(experiment_path):
        """
        Loads scalar and text data from TensorBoard event files.
        """
        event_file = glob.glob(os.path.join(experiment_path, 'tensorboard_logs', 'events.out.tfevents.*'))
        if not event_file:
            return {}, ""

        ea = event_accumulator.EventAccumulator(
            event_file[0],
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
            }
        )
        ea.Reload()

        # --- Load Scalars ---
        scalar_tags = ea.Tags()['scalars']
        dataframes = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            df = pd.DataFrame({'step': steps, 'value': values, 'tag': tag})
            
            if tag.startswith('Losses'): group, metric = 'Losses', tag.split('/')[-1]
            elif tag.startswith('Hormone_Levels'): group, metric = 'Hormone Levels', tag.split('/')[-1]
            elif tag.startswith('Physics_Parameters'): group, metric = 'Physics Parameters', tag.split('/')[-1]
            elif tag == 'Learning_Rate': group, metric = 'Learning Rate', 'Learning Rate'
            else: group, metric = 'Other', tag
            
            df['metric'] = metric
            if group not in dataframes: dataframes[group] = []
            dataframes[group].append(df)

        for group, df_list in dataframes.items():
            dataframes[group] = pd.concat(df_list, ignore_index=True)

        # --- Load Text (Cluster Analysis) ---
        cluster_analysis_text = ""
        if 'tensors' in ea.Tags(): # Text is stored as tensors
            if 'Cluster_Analysis' in ea.Tags()['tensors']:
                text_events = ea.Tensors('Cluster_Analysis')
                if text_events:
                    # Get the most recent text event
                    latest_text_event = text_events[-1]
                    cluster_analysis_text = latest_text_event.tensor_proto.string_val[0].decode('utf-8')

        return dataframes, cluster_analysis_text
    
    @staticmethod
    def _get_experiment_dirs():
        """Finds all experiment directories."""
        experiments_path = "experiments"
        if not os.path.isdir(experiments_path):
            return []
        # Sort directories by modification time, newest first
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

        # --- Sidebar for Experiment Selection ---
        st.sidebar.header("Experiment Controls")
        experiment_dirs = self._get_experiment_dirs()

        if not experiment_dirs:
            st.error("No experiment directories found in the 'experiments' folder.")
            st.stop()

        selected_experiment_id = st.sidebar.selectbox(
            "Select an Experiment ID",
            options=experiment_dirs,
            index=0 # The list is now sorted, so the first one is the most recent
        )

        selected_experiment_path = os.path.join("experiments", selected_experiment_id)

        if st.sidebar.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.sidebar.markdown("---")
        config = self._load_config(selected_experiment_path)
        if config:
            st.sidebar.subheader("Experiment Config")
            st.sidebar.json(config, expanded=False)

        # --- Main Dashboard Area ---
        if selected_experiment_id:
            st.header(f"📊 Analysis for Experiment: `{selected_experiment_id}`")

            with st.spinner("Loading TensorBoard data... This may take a moment for large experiments."):
                all_data, cluster_text = self._load_tensorboard_data(selected_experiment_path)

            if not all_data:
                st.warning("No TensorBoard data found for this experiment yet. Please wait for the training to start.")
                st.stop()

            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Key Metrics",
                "🌌 2D Space Evolution",
                "🧬 8D Cluster Analysis",
                "⚡ Embedding Projector"
            ])

            with tab1:
                st.subheader("Training Performance")
                col1, col2 = st.columns(2)
                with col1:
                    if 'Losses' in all_data:
                        st.altair_chart(self._plot_chart(all_data['Losses'], 'Losses Over Time'), use_container_width=True)
                    else: st.info("Loss data not available.")
                with col2:
                    if 'Learning Rate' in all_data:
                        st.altair_chart(self._plot_chart(all_data['Learning Rate'], 'Learning Rate Schedule'), use_container_width=True)
                    else: st.info("Learning Rate data not available.")
                
                st.subheader("Learnable Physics Parameters")
                if 'Physics Parameters' in all_data:
                     st.altair_chart(self._plot_chart(all_data['Physics Parameters'], 'Physics Parameter Evolution'), use_container_width=True)
                else: st.info("Physics parameter data not available.")

            with tab2:
                st.subheader("Evolution of 2D Semantic Space")
                frames_path = os.path.join(selected_experiment_path, "semantic_space_frames")
                if os.path.isdir(frames_path) and glob.glob(os.path.join(frames_path, "frame_*.png")):
                    frame_files = sorted(glob.glob(os.path.join(frames_path, "frame_*.png")))
                    
                    def get_step_from_frame(f):
                        match = re.search(r'frame_(\d+).png', os.path.basename(f))
                        return int(match.group(1)) if match else 0

                    frame_steps = [get_step_from_frame(f) for f in frame_files]

                    selected_step = st.select_slider(
                        "Scrub through training frames (by global step):",
                        options=frame_steps,
                        value=frame_steps[-1]
                    )
                    selected_frame_path = os.path.join(frames_path, f'frame_{selected_step:06d}.png')
                    st.image(selected_frame_path)
                else:
                    st.info("No semantic space visualizations found for this experiment.")

            with tab3:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Semantic Endocrine System")
                    if 'Hormone Levels' in all_data:
                        st.altair_chart(self._plot_chart(all_data['Hormone Levels'], 'Hormone Levels Over Time'), use_container_width=True)
                    else:
                        st.info("Hormone level data not available.")
                
                with col2:
                    st.subheader("8D Cluster Centroid Analysis")
                    if cluster_text:
                        st.markdown(cluster_text)
                    else:
                        st.info("Cluster analysis text not yet available. It is logged periodically during training.")

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
