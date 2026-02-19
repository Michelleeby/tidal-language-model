import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime
from ruamel.yaml import YAML

# Fix Plotly/Streamlit interaction issue - reset default template
# Importing Streamlit can alter Plotly's default theme, causing rendering issues
pio.templates.default = 'plotly_dark'

st.set_page_config(
    page_title="Tidal Language Model Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: transparent;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def find_experiments(base_dir="experiments"):
    """Find all experiment directories."""
    if not os.path.exists(base_dir):
        return []

    experiments = []
    for item in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, item)
        metrics_dir = os.path.join(exp_path, "dashboard_metrics")
        tensorboard_dir = os.path.join(exp_path, "tensorboard_logs")
        # Accept experiments with either dashboard_metrics or tensorboard_logs
        if os.path.isdir(exp_path) and (os.path.exists(metrics_dir) or os.path.exists(tensorboard_dir)):
            experiments.append(item)

    return sorted(experiments, reverse=True)  # Most recent first


def read_last_n_lines(filepath, n):
    """Efficiently read the last N lines of a file using reverse seeking.

    Returns:
        tuple: (lines_list, total_lines_estimate)
    """
    with open(filepath, 'rb') as f:
        # Get file size
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()

        if file_size == 0:
            return [], 0

        # Start with a reasonable chunk size, grow if needed
        chunk_size = min(file_size, n * 500)  # Estimate ~500 bytes per line
        lines = []
        position = file_size

        while len(lines) <= n and position > 0:
            # Move back by chunk_size
            read_size = min(chunk_size, position)
            position -= read_size
            f.seek(position)
            chunk = f.read(read_size)

            # Decode and split into lines
            try:
                chunk_text = chunk.decode('utf-8')
            except UnicodeDecodeError:
                # Handle partial UTF-8 at chunk boundary
                position += 1
                f.seek(position)
                chunk = f.read(read_size - 1)
                chunk_text = chunk.decode('utf-8', errors='ignore')

            chunk_lines = chunk_text.split('\n')

            # Combine with previously read lines
            if lines:
                # First line of previous chunk may be partial
                chunk_lines[-1] += lines[0]
                lines = chunk_lines + lines[1:]
            else:
                lines = chunk_lines

            # Increase chunk size for next iteration
            chunk_size = min(chunk_size * 2, 10 * 1024 * 1024)  # Max 10MB chunks

        # Filter empty lines and take last n
        lines = [line for line in lines if line.strip()]
        total_estimate = len(lines) if position == 0 else int(file_size / (file_size - position + 1) * len(lines))

        return lines[-n:], max(total_estimate, len(lines))


@st.cache_data(ttl=60)
def load_metrics_windowed(metrics_file, window_size=10000, _mtime=None):
    """Load only the last N metrics from JSONL file using efficient tail reading.

    Args:
        metrics_file: Path to the metrics file
        window_size: Number of lines to read from the end
        _mtime: Modification time (used for cache invalidation)

    Returns:
        tuple: (metrics_list, total_count)
    """
    if not os.path.exists(metrics_file):
        return [], 0

    try:
        lines, total_estimate = read_last_n_lines(metrics_file, window_size)
        metrics = [json.loads(line) for line in lines if line.strip()]
        return metrics, total_estimate
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return [], 0


@st.cache_data(ttl=300)
def load_metrics_downsampled(metrics_file, max_points=5000, _mtime=None):
    """Load metrics with strided sampling for historical overview.

    Uses file size estimation to avoid counting all lines.

    Args:
        metrics_file: Path to the metrics file
        max_points: Maximum number of points to return
        _mtime: Modification time (used for cache invalidation)

    Returns:
        tuple: (metrics_list, total_count)
    """
    if not os.path.exists(metrics_file):
        return [], 0

    try:
        file_size = os.path.getsize(metrics_file)

        # Sample first 100 lines to estimate average line size
        sample_lines = []
        sample_bytes = 0
        with open(metrics_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                if line.strip():
                    sample_lines.append(line)
                    sample_bytes += len(line.encode('utf-8'))

        if not sample_lines:
            return [], 0

        avg_line_size = sample_bytes / len(sample_lines)
        total_estimate = int(file_size / avg_line_size)

        if total_estimate <= max_points:
            # Small file - read all
            metrics = [json.loads(line) for line in sample_lines]
            with open(metrics_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 100 and line.strip():  # Skip already-read lines
                        metrics.append(json.loads(line))
            return metrics, len(metrics)

        # Calculate stride for sampling
        stride = max(1, total_estimate // max_points)

        metrics = []
        with open(metrics_file, 'r') as f:
            for i, line in enumerate(f):
                if i % stride == 0 and line.strip():
                    metrics.append(json.loads(line))
                    if len(metrics) >= max_points:
                        break

        # Always include last few points for recent data
        last_lines, _ = read_last_n_lines(metrics_file, 10)
        for line in last_lines:
            if line.strip():
                metrics.append(json.loads(line))

        return metrics, total_estimate
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return [], 0


@st.cache_data(ttl=15)
def load_metrics(metrics_file, _mtime=None):
    """Load metrics from JSONL file (legacy function for compatibility).

    Args:
        metrics_file: Path to the metrics file
        _mtime: Modification time (used for cache invalidation)
    """
    if not os.path.exists(metrics_file):
        return []

    metrics = []
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

    return metrics


def load_status(status_file):
    """Load training status."""
    if not os.path.exists(status_file):
        return {"status": "unknown"}

    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading status: {e}")
        return {"status": "error"}


def load_tensorboard_metrics(tensorboard_dir):
    """Load metrics from TensorBoard event files.

    Handles two layouts:
    1. Subdirectory layout: scalars split across subdirs (e.g. Loss_Total/,
       Loss_Prediction/, Loss_Physics/) with a generic tag like 'Loss' in each,
       plus root-level tags like 'Learning' for learning rate.
    2. Flat layout: all tags in the root EventAccumulator with descriptive names.

    Args:
        tensorboard_dir: Path to the tensorboard_logs directory

    Returns:
        tuple: (list of metric dicts, total count)
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        st.warning("TensorBoard not installed. Cannot read legacy experiment data.")
        return [], 0

    if not os.path.exists(tensorboard_dir):
        return [], 0

    # Mapping from (subdir_name, tag) to dashboard column name
    SUBDIR_TAG_MAP = {
        ('Loss_Total', 'Loss'): 'Losses/Total',
        ('Loss_Prediction', 'Loss'): 'Losses/Prediction',
        ('Loss_Physics', 'Loss'): 'Losses/Physics',
    }

    # Mapping from root-level tag names to dashboard column names
    ROOT_TAG_MAP = {
        'Learning': 'Learning Rate',
        'Learning_Rate': 'Learning Rate',
    }

    def _map_tag_heuristic(tag):
        """Fallback heuristic mapping for tags not in the explicit maps."""
        tag_lower = tag.lower().replace('/', '_').replace(' ', '_')
        if 'loss' in tag_lower:
            if 'total' in tag_lower or tag == 'Loss/Total':
                return 'Losses/Total'
            elif 'prediction' in tag_lower or 'pred' in tag_lower:
                return 'Losses/Prediction'
            elif 'physics' in tag_lower:
                return 'Losses/Physics'
        elif 'learning_rate' in tag_lower:
            return 'Learning Rate'
        elif 'f_pos' in tag_lower or 'free_energy_pos' in tag_lower:
            return 'Energy/F_pos'
        elif 'f_neg' in tag_lower or 'free_energy_neg' in tag_lower:
            return 'Energy/F_neg'
        elif 'hormone' in tag_lower:
            if 'catalyst' in tag_lower:
                return 'Hormones/catalyst_hormone'
            elif 'stress' in tag_lower:
                return 'Hormones/stress_hormone'
            elif 'inhibitor' in tag_lower:
                return 'Hormones/inhibitor_hormone'
        elif tag_lower == 'g' or 'gravitational' in tag_lower:
            return 'Physics/G'
        elif 'repulsion' in tag_lower:
            return 'Physics/Repulsion_Strength'
        elif 'well' in tag_lower and 'attraction' in tag_lower:
            return 'Physics/Well_Attraction'
        elif 'temperature' in tag_lower and 'physics' in tag_lower:
            return 'Physics/Temperature'
        return None

    def _ingest_events(ea, column_name, metrics_by_step, tag):
        """Add events from an EventAccumulator tag into metrics_by_step."""
        for event in ea.Scalars(tag):
            step = event.step
            if step not in metrics_by_step:
                metrics_by_step[step] = {'step': step}
            metrics_by_step[step][column_name] = event.value

    try:
        metrics_by_step = {}

        # 1. Load root-level EventAccumulator
        ea = EventAccumulator(tensorboard_dir)
        ea.Reload()
        for tag in ea.Tags().get('scalars', []):
            col = ROOT_TAG_MAP.get(tag) or _map_tag_heuristic(tag)
            if col:
                _ingest_events(ea, col, metrics_by_step, tag)

        # 2. Scan subdirectories for additional EventAccumulators
        for entry in os.listdir(tensorboard_dir):
            subdir_path = os.path.join(tensorboard_dir, entry)
            if not os.path.isdir(subdir_path):
                continue
            try:
                sub_ea = EventAccumulator(subdir_path)
                sub_ea.Reload()
                for tag in sub_ea.Tags().get('scalars', []):
                    col = SUBDIR_TAG_MAP.get((entry, tag)) or _map_tag_heuristic(f"{entry}/{tag}")
                    if col:
                        _ingest_events(sub_ea, col, metrics_by_step, tag)
            except Exception:
                continue

        if not metrics_by_step:
            return [], 0

        # Convert to sorted list
        metrics = [metrics_by_step[step] for step in sorted(metrics_by_step.keys())]
        return metrics, len(metrics)

    except Exception as e:
        st.warning(f"Error reading TensorBoard data: {e}")
        return [], 0


@st.cache_data(ttl=30)
def load_semantic_space(semantic_file, _mtime=None):
    """Load semantic space data.

    Args:
        semantic_file: Path to the semantic space file
        _mtime: Modification time (used for cache invalidation)
    """
    if not os.path.exists(semantic_file):
        return None

    try:
        with open(semantic_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading semantic space: {e}")
        return None


@st.cache_data(ttl=60)
def load_trigger_events(trigger_file, last_n=100, _mtime=None):
    """Load trigger events from JSONL file (only last N lines).

    Uses efficient tail reading to avoid scanning entire file.

    Args:
        trigger_file: Path to the trigger events file
        last_n: Number of most recent events to load
        _mtime: Modification time (used for cache invalidation)
    """
    if not os.path.exists(trigger_file):
        return [], 0

    try:
        lines, total_estimate = read_last_n_lines(trigger_file, last_n)
        events = [json.loads(line) for line in lines if line.strip()]
        return events, total_estimate
    except Exception as e:
        st.error(f"Error loading trigger events: {e}")
        return [], 0


def load_config(exp_dir):
    """Load configuration from experiment directory to detect model type."""
    config_file = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(config_file):
        return None

    try:
        yaml = YAML(typ='safe')
        with open(config_file, 'r') as f:
            return yaml.load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


def get_model_type(exp_dir):
    """Determine the model type (tidal or constant) from the experiment config."""
    config = load_config(exp_dir)
    if config:
        return config.get("MODEL_TYPE", "tidal")
    return "tidal"  # Default to tidal if config not found


def get_scatter_type(data_length, threshold=5000):
    """Return Scattergl for large datasets, Scatter for small ones.

    WebGL rendering (Scattergl) is much faster for large datasets but has
    slightly less crisp rendering for small datasets.
    """
    return go.Scattergl if data_length > threshold else go.Scatter


def create_2x2_subplot_figure(df, subplot_specs, chart_key_prefix="chart"):
    """Create a 2x2 subplot figure with synchronized crosshair tooltips.

    Args:
        df: DataFrame with a 'step' column and metric columns.
        subplot_specs: List of 4 dicts, each with:
            - "column": column name in df (for single-trace subplots)
            - "title": subplot title
            - "color": line color (for single-trace subplots)
            - "y_label": y-axis label
            - "traces": optional list of dicts for multi-trace subplots, each with:
                - "column": column name
                - "name": trace display name
                - "color": line color
        chart_key_prefix: prefix for Streamlit chart key (unused but kept for API)

    Returns:
        go.Figure or None if all subplots are empty.
    """
    if df.empty or "step" not in df.columns:
        return None

    # Build subplot titles
    titles = [spec["title"] for spec in subplot_specs]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        shared_xaxes='all',
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    data_length = len(df)
    ScatterType = get_scatter_type(data_length)
    any_data = False

    for i, spec in enumerate(subplot_specs):
        row = (i // 2) + 1
        col = (i % 2) + 1

        traces = spec.get("traces")
        if traces:
            # Multi-trace subplot
            subplot_has_data = False
            for trace_spec in traces:
                col_name = trace_spec["column"]
                if col_name in df.columns and not df[col_name].isna().all():
                    fig.add_trace(
                        ScatterType(
                            x=df["step"],
                            y=df[col_name],
                            name=trace_spec["name"],
                            line=dict(color=trace_spec["color"], width=2),
                            mode='lines',
                            showlegend=True,
                        ),
                        row=row, col=col,
                    )
                    subplot_has_data = True
            if subplot_has_data:
                any_data = True
            else:
                fig.add_annotation(
                    text="No data available",
                    xref=f"x{i + 1}" if i > 0 else "x",
                    yref=f"y{i + 1}" if i > 0 else "y",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    showarrow=False,
                    font=dict(size=14, color="#888888"),
                )
        else:
            # Single-trace subplot
            col_name = spec.get("column", "")
            if col_name and col_name in df.columns and not df[col_name].isna().all():
                fig.add_trace(
                    ScatterType(
                        x=df["step"],
                        y=df[col_name],
                        name=spec["title"],
                        line=dict(color=spec["color"], width=2),
                        mode='lines',
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
                any_data = True
            else:
                fig.add_annotation(
                    text="No data available",
                    xref=f"x{i + 1}" if i > 0 else "x",
                    yref=f"y{i + 1}" if i > 0 else "y",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    showarrow=False,
                    font=dict(size=14, color="#888888"),
                )

    if not any_data:
        return None

    # Configure axes: spike lines on all x-axes, y-labels, step labels on bottom row
    spike_props = dict(
        showspikes=True,
        spikemode='across',
        spikethickness=1,
        spikecolor='#888888',
        spikedash='dot',
    )

    for i, spec in enumerate(subplot_specs):
        axis_suffix = str(i + 1) if i > 0 else ""
        x_axis_key = f"xaxis{axis_suffix}"
        y_axis_key = f"yaxis{axis_suffix}"

        row = (i // 2) + 1
        # X-axis: spike lines on all, "Step" label only on bottom row
        x_update = dict(**spike_props)
        if row == 2:
            x_update["title"] = "Step"
        fig.update_layout(**{x_axis_key: x_update})

        # Y-axis label
        fig.update_layout(**{y_axis_key: dict(title=spec.get("y_label", ""))})

    fig.update_layout(
        height=750,
        template='plotly_dark',
        hovermode='x',
        hoversubplots='axis',
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig



def plot_hormone_levels(df):
    """Plot hormone levels over time."""
    if df.empty or "step" not in df.columns:
        return None

    hormone_cols = [col for col in df.columns if col.startswith("Hormones/")]

    if not hormone_cols:
        return None

    # Check if any hormone column has data
    has_data = any(not df[col].isna().all() for col in hormone_cols)
    if not has_data:
        return None

    ScatterType = get_scatter_type(len(df))
    fig = go.Figure()

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, col in enumerate(hormone_cols):
        if not df[col].isna().all():  # Only plot if column has data
            hormone_name = col.split("/")[1].replace("_", " ").title()
            fig.add_trace(ScatterType(
                x=df["step"],
                y=df[col],
                name=hormone_name,
                line=dict(color=colors[i % len(colors)], width=2),
                mode='lines'
            ))

    fig.update_layout(
        title="Hormone Levels Over Time",
        xaxis_title="Step",
        yaxis_title="Hormone Level",
        height=400,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=50, r=20, t=40, b=40)
    )

    return fig


def plot_semantic_space_2d(semantic_data, vocab_map=None):
    """Plot 2D semantic space."""
    if not semantic_data or "positions_2d" not in semantic_data:
        return None

    positions = np.array(semantic_data["positions_2d"])

    # Create hover text
    hover_text = []
    if "token_ids" in semantic_data and vocab_map:
        idx_to_word = {v: k for k, v in vocab_map.items()}
        for token_id in semantic_data["token_ids"]:
            word = idx_to_word.get(token_id, f"ID:{token_id}")
            hover_text.append(word)
    else:
        hover_text = [f"Point {i}" for i in range(len(positions))]

    # Color by cluster or magnitude
    colors = np.linalg.norm(positions, axis=1) if len(positions) > 0 else []

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Magnitude"),
            line=dict(width=0.5, color='white')
        ),
        text=hover_text,
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))

    # Add forces as arrows if available
    if "forces_2d" in semantic_data:
        forces = np.array(semantic_data["forces_2d"])
        # Sample forces to avoid overcrowding
        sample_idx = np.linspace(0, len(forces)-1, min(50, len(forces)), dtype=int)

        for idx in sample_idx:
            fig.add_annotation(
                x=positions[idx, 0],
                y=positions[idx, 1],
                ax=positions[idx, 0] + forces[idx, 0] * 0.01,
                ay=positions[idx, 1] + forces[idx, 1] * 0.01,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='rgba(255,0,0,0.3)'
            )

    fig.update_layout(
        title="2D Physical Space (with Physics Simulation)",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        height=600,
        hovermode='closest',
        template='plotly_dark'
    )

    return fig


def plot_semantic_space_8d(semantic_data, config=None):
    """Plot 8D semantic space using dimensionality reduction."""
    if not semantic_data or "positions_8d" not in semantic_data:
        return None

    positions_8d = np.array(semantic_data["positions_8d"])

    # Use first 3 principal dimensions for 3D visualization
    # In a real implementation, you might want to use PCA or t-SNE
    # For now, just use the first 3 dimensions

    if positions_8d.shape[1] < 3:
        return None

    fig = go.Figure(data=[go.Scatter3d(
        x=positions_8d[:, 0],  # G-Axis (Groundedness)
        y=positions_8d[:, 2],  # V-Axis (Valence)
        z=positions_8d[:, 3],  # A-Axis (Arousal)
        mode='markers',
        marker=dict(
            size=5,
            color=positions_8d[:, 1],  # Color by X-Axis (Taxonomic Specificity)
            colorscale='Rainbow',
            showscale=True,
            colorbar=dict(title="Taxonomic<br>Specificity"),
            line=dict(width=0.5, color='white')
        ),
        hovertemplate='<b>Point</b><br>G: %{x:.3f}<br>V: %{y:.3f}<br>A: %{z:.3f}<extra></extra>'
    )])

    fig.update_layout(
        title="8D Semantic Space (G-V-A Projection)",
        scene=dict(
            xaxis_title="Groundedness (Concrete ↔ Abstract)",
            yaxis_title="Valence (Negative ↔ Positive)",
            zaxis_title="Arousal (Calm ↔ Excited)",
        ),
        height=700,
        template='plotly_dark'
    )

    return fig


def render_trigger_events_table(events):
    """Render trigger events as a table."""
    if not events:
        st.info("No trigger events recorded yet.")
        return

    # Take last 100 events
    recent_events = events[-100:]

    # Create DataFrame
    df = pd.DataFrame(recent_events)

    # Format columns if they exist
    if "strength" in df.columns:
        df["strength"] = df["strength"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    if "threshold" in df.columns:
        df["threshold"] = df["threshold"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    if "tidal_level" in df.columns:
        df["tidal_level"] = df["tidal_level"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    # Display most recent first
    df = df.sort_values("step", ascending=False)

    st.dataframe(df, width='stretch', height=400)


def render_status_display(status):
    """Render the training status display."""
    status_emoji = {
        "initialized": "🟡",
        "training": "🟢",
        "completed": "🔵",
        "error": "🔴",
        "unknown": "⚪"
    }

    current_status = status.get("status", "unknown")
    st.sidebar.markdown(f"{status_emoji.get(current_status, '⚪')} **{current_status.upper()}**")

    if "current_step" in status:
        st.sidebar.metric("Current Step", f"{status['current_step']:,}")

    if "last_update" in status:
        last_update = datetime.fromtimestamp(status["last_update"])
        st.sidebar.text(f"Last Update: {last_update.strftime('%H:%M:%S')}")


def render_loss_metrics_tab(df):
    """Render Tab 1: Loss & Metrics."""
    st.header("Training Loss Curves")

    # Key metrics at the top - only show if we have data
    if len(df) > 0:
        latest = df.iloc[-1]

        # Collect available metrics (skip NaN values)
        metrics_to_show = []
        if "Losses/Total" in latest and not pd.isna(latest['Losses/Total']):
            metrics_to_show.append(("Total Loss", f"{latest['Losses/Total']:.4f}"))
        if "Losses/Prediction" in latest and not pd.isna(latest['Losses/Prediction']):
            metrics_to_show.append(("Prediction Loss", f"{latest['Losses/Prediction']:.4f}"))
        if "Losses/Physics" in latest and not pd.isna(latest['Losses/Physics']):
            metrics_to_show.append(("Physics Loss", f"{latest['Losses/Physics']:.4f}"))
        if "Learning Rate" in latest and not pd.isna(latest['Learning Rate']):
            metrics_to_show.append(("Learning Rate", f"{latest['Learning Rate']:.2e}"))

        # Only create columns if we have metrics to show
        if metrics_to_show:
            cols = st.columns(len(metrics_to_show))
            for i, (label, value) in enumerate(metrics_to_show):
                with cols[i]:
                    st.metric(label, value)

    # Loss plots in synchronized 2x2 grid
    subplot_specs = [
        {"column": "Losses/Total", "title": "Total Loss", "color": "#1f77b4", "y_label": "Loss"},
        {"column": "Losses/Prediction", "title": "Prediction Loss", "color": "#ff7f0e", "y_label": "Loss"},
        {"column": "Losses/Physics", "title": "Physics Loss", "color": "#2ca02c", "y_label": "Loss"},
        {"title": "Free Energy Components", "y_label": "Free Energy", "traces": [
            {"column": "Energy/F_pos", "name": "F_pos (Target)", "color": "#d62728"},
            {"column": "Energy/F_neg", "name": "F_neg (Negative)", "color": "#9467bd"},
        ]},
    ]
    fig = create_2x2_subplot_figure(df, subplot_specs)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="chart_loss_grid")
    else:
        st.info("No loss data available yet.")

    # Recent metrics table
    with st.expander("Recent Metrics (Last 20 steps)"):
        recent_df = df.tail(20).copy()
        for col in recent_df.select_dtypes(include=[np.number]).columns:
            if col != "step":
                recent_df[col] = recent_df[col].round(6)
        st.dataframe(recent_df, width='stretch')


def render_constant_model_tab(df):
    """Render metrics tab for the constant (baseline) model.

    Shows Total Loss, Prediction Loss, Physics Loss, and Learning Rate plots.
    """
    st.header("Training Loss & Learning Rate")

    # Key metrics at the top
    if len(df) > 0:
        latest = df.iloc[-1]
        metrics_to_show = []
        if "Losses/Total" in latest and not pd.isna(latest['Losses/Total']):
            metrics_to_show.append(("Total Loss", f"{latest['Losses/Total']:.4f}"))
        if "Losses/Prediction" in latest and not pd.isna(latest['Losses/Prediction']):
            metrics_to_show.append(("Prediction Loss", f"{latest['Losses/Prediction']:.4f}"))
        if "Losses/Physics" in latest and not pd.isna(latest['Losses/Physics']):
            metrics_to_show.append(("Physics Loss", f"{latest['Losses/Physics']:.4f}"))
        if "Learning Rate" in latest and not pd.isna(latest['Learning Rate']):
            metrics_to_show.append(("Learning Rate", f"{latest['Learning Rate']:.2e}"))

        if metrics_to_show:
            cols = st.columns(len(metrics_to_show))
            for i, (label, value) in enumerate(metrics_to_show):
                with cols[i]:
                    st.metric(label, value)

    # Plots in synchronized 2x2 grid
    subplot_specs = [
        {"column": "Losses/Total", "title": "Total Loss", "color": "#1f77b4", "y_label": "Loss"},
        {"column": "Losses/Prediction", "title": "Prediction Loss", "color": "#ff7f0e", "y_label": "Loss"},
        {"column": "Losses/Physics", "title": "Physics Loss", "color": "#2ca02c", "y_label": "Loss"},
        {"column": "Learning Rate", "title": "Learning Rate", "color": "#9467bd", "y_label": "Learning Rate"},
    ]
    fig = create_2x2_subplot_figure(df, subplot_specs)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="chart_const_grid")
    else:
        st.info("No training data available yet.")

    # Recent metrics table
    with st.expander("📋 Recent Metrics (Last 20 steps)"):
        recent_df = df.tail(20).copy()
        for col in recent_df.select_dtypes(include=[np.number]).columns:
            if col != "step":
                recent_df[col] = recent_df[col].round(6)
        st.dataframe(recent_df, width='stretch')


def render_physics_tab(df):
    """Render Tab 2: Physics Parameters."""
    st.header("Physics Simulation Parameters")

    # Current values
    if len(df) > 0:
        latest = df.iloc[-1]

        # Collect available physics metrics (skip NaN values)
        physics_metrics = []
        if "Physics/G" in latest and not pd.isna(latest['Physics/G']):
            physics_metrics.append(("Gravitational Constant", f"{latest['Physics/G']:.4f}"))
        if "Physics/Repulsion_Strength" in latest and not pd.isna(latest['Physics/Repulsion_Strength']):
            physics_metrics.append(("Repulsion Strength", f"{latest['Physics/Repulsion_Strength']:.4f}"))
        if "Physics/Well_Attraction" in latest and not pd.isna(latest['Physics/Well_Attraction']):
            physics_metrics.append(("Well Attraction", f"{latest['Physics/Well_Attraction']:.4f}"))
        if "Physics/Temperature" in latest and not pd.isna(latest['Physics/Temperature']):
            physics_metrics.append(("Temperature", f"{latest['Physics/Temperature']:.4f}"))

        # Only create columns if we have metrics to show
        if physics_metrics:
            cols = st.columns(len(physics_metrics))
            for i, (label, value) in enumerate(physics_metrics):
                with cols[i]:
                    st.metric(label, value)

    # Physics parameter evolution in synchronized 2x2 grid
    subplot_specs = [
        {"column": "Physics/G", "title": "Gravitational Constant (G)", "color": "#1f77b4", "y_label": "Value"},
        {"column": "Physics/Repulsion_Strength", "title": "Repulsion Strength", "color": "#ff7f0e", "y_label": "Value"},
        {"column": "Physics/Well_Attraction", "title": "Well Attraction", "color": "#2ca02c", "y_label": "Value"},
        {"column": "Physics/Temperature", "title": "Temperature", "color": "#d62728", "y_label": "Value"},
    ]
    fig = create_2x2_subplot_figure(df, subplot_specs)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="chart_physics_grid")


def render_endocrine_tab(df):
    """Render Tab 3: Endocrine System."""
    st.header("Semantic Endocrine System")

    # Current hormone levels
    if len(df) > 0:
        latest = df.iloc[-1]
        hormone_cols = [col for col in df.columns if col.startswith("Hormones/")]

        if hormone_cols:
            valid_hormones = [(col_name, col_name.split("/")[1].replace("_", " ").title())
                              for col_name in hormone_cols
                              if not pd.isna(latest.get(col_name, float('nan')))]
            if valid_hormones:
                cols = st.columns(len(valid_hormones))
                for i, (col_name, hormone_name) in enumerate(valid_hormones):
                    with cols[i]:
                        st.metric(hormone_name, f"{latest[col_name]:.4f}")
        else:
            st.info("Endocrine system is not enabled or no hormone data available.")

    # Hormone level evolution
    hormone_fig = plot_hormone_levels(df)
    if hormone_fig:
        st.plotly_chart(hormone_fig, width='stretch', key="chart_hormones")
    else:
        st.info("No hormone level data available yet.")


def render_semantic_universe_tab(df, semantic_file):
    """Render Tab 4: Semantic Universe."""
    st.header("Semantic Space Visualization")

    semantic_mtime = os.path.getmtime(semantic_file) if os.path.exists(semantic_file) else None
    semantic_data = load_semantic_space(semantic_file, _mtime=semantic_mtime)

    if semantic_data:
        st.markdown(f"**Step:** {semantic_data.get('step', 'N/A')}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("2D Physical Space")
            fig_2d = plot_semantic_space_2d(semantic_data)
            if fig_2d:
                st.plotly_chart(fig_2d, width='stretch', key="chart_semantic_2d")
            else:
                st.info("No 2D space data available.")

        with col2:
            st.subheader("8D Semantic Space")
            fig_8d = plot_semantic_space_8d(semantic_data)
            if fig_8d:
                st.plotly_chart(fig_8d, width='stretch', key="chart_semantic_8d")
            else:
                st.info("No 8D space data available.")

        # Additional info
        with st.expander("ℹ️ Semantic Space Info"):
            st.markdown("""
            **2D Physical Space:** Shows word embeddings after physics simulation with gravitational forces.
            Red arrows indicate force vectors acting on sampled points.

            **8D Semantic Space:** Four conceptual planes:
            - **Core Conceptual** (G, X): Groundedness & Taxonomic Specificity
            - **Affective** (V, A): Valence & Arousal
            - **Interoceptive** (H, S): Homeostasis & Somatic Focus
            - **Structural** (F, T): Functional Role & Temporal Orientation
            """)
    else:
        st.info("No semantic space data available yet. Visualizations will appear during training.")


def render_trigger_events_tab(trigger_file):
    """Render Tab 5: Trigger Events."""
    st.header("Endocrine Trigger Events")

    trigger_mtime = os.path.getmtime(trigger_file) if os.path.exists(trigger_file) else None
    trigger_events, total_count = load_trigger_events(trigger_file, last_n=100, _mtime=trigger_mtime)

    if trigger_events:
        st.markdown(f"**Showing last {len(trigger_events)} of {total_count:,} total events**")
        render_trigger_events_table(trigger_events)
    else:
        st.info("No trigger events recorded yet.")


@st.fragment(run_every=45)
def auto_refresh_data_and_plots(exp_dir, model_type, view_mode, window_size, max_points):
    """Fragment that auto-refreshes every 45s without re-rendering the entire page.

    Args:
        exp_dir: Path to the experiment directory
        model_type: Either "tidal" or "constant"
        view_mode: Either "Recent" or "Historical (downsampled)"
        window_size: Number of recent points to show in Recent mode
        max_points: Maximum points to show in Historical mode
    """
    metrics_dir = os.path.join(exp_dir, "dashboard_metrics")
    metrics_file = os.path.join(metrics_dir, "metrics.jsonl")
    semantic_file = os.path.join(metrics_dir, "semantic_space.json")
    trigger_file = os.path.join(metrics_dir, "trigger_events.jsonl")
    tensorboard_dir = os.path.join(exp_dir, "tensorboard_logs")

    # Check if auto-refresh is paused
    paused = st.session_state.get('auto_refresh_paused', False)

    # Compute lightweight fingerprint for data-change gating
    metrics_mtime = os.path.getmtime(metrics_file) if os.path.exists(metrics_file) else 0
    metrics_size = os.path.getsize(metrics_file) if os.path.exists(metrics_file) else 0
    fingerprint = (metrics_mtime, metrics_size, view_mode, window_size, max_points, exp_dir)
    data_changed = fingerprint != st.session_state.get('_data_fingerprint')

    if data_changed and not paused:
        st.session_state._data_fingerprint = fingerprint

        # Load metrics - try JSONL first, fall back to TensorBoard
        metrics = []
        total_count = 0
        using_tensorboard = False

        if os.path.exists(metrics_file):
            if view_mode == "Recent":
                metrics, total_count = load_metrics_windowed(metrics_file, window_size=window_size, _mtime=metrics_mtime)
            else:
                metrics, total_count = load_metrics_downsampled(metrics_file, max_points=max_points, _mtime=metrics_mtime)
        elif os.path.exists(tensorboard_dir):
            # Fall back to TensorBoard data for older experiments
            metrics, total_count = load_tensorboard_metrics(tensorboard_dir)
            using_tensorboard = True

        if metrics:
            st.session_state._cached_df = pd.DataFrame(metrics)
            st.session_state._cached_total_count = total_count
            st.session_state._cached_using_tensorboard = using_tensorboard
        else:
            st.session_state._cached_df = None
            st.session_state._cached_total_count = 0
            st.session_state._cached_using_tensorboard = False

    # Use cached data for rendering
    df = st.session_state.get('_cached_df')
    total_count = st.session_state.get('_cached_total_count', 0)
    using_tensorboard = st.session_state.get('_cached_using_tensorboard', False)

    if df is None or df.empty:
        st.warning("No metrics data available yet. Training may not have started.")
        return

    if using_tensorboard:
        st.info("Showing data from TensorBoard logs (legacy format)")

    # Show data coverage info
    if total_count > 0:
        coverage_pct = (len(df) / total_count) * 100
        if view_mode == "Recent":
            st.caption(f"Showing last {len(df):,} of {total_count:,} total points ({coverage_pct:.1f}%)")
        else:
            st.caption(f"Showing {len(df):,} downsampled points from {total_count:,} total ({coverage_pct:.1f}%)")

    # Create tabs inside the fragment so they replace on each run
    if model_type == "constant":
        tabs = st.tabs(["📊 Loss & Learning Rate"])
        with tabs[0]:
            render_constant_model_tab(df)
    else:
        tabs = st.tabs([
            "📊 Loss & Metrics",
            "⚛️ Physics Parameters",
            "🧬 Endocrine System",
            "🌌 Semantic Universe",
            "⚡ Trigger Events"
        ])

        with tabs[0]:
            render_loss_metrics_tab(df)

        with tabs[1]:
            render_physics_tab(df)

        with tabs[2]:
            render_endocrine_tab(df)

        with tabs[3]:
            render_semantic_universe_tab(df, semantic_file)

        with tabs[4]:
            render_trigger_events_tab(trigger_file)


def main():
    st.markdown('<div class="main-header">🌊 Language Model Training Dashboard</div>', unsafe_allow_html=True)

    # Initialize session state for status placeholder
    if 'status_placeholder' not in st.session_state:
        st.session_state.status_placeholder = None

    # Sidebar - Experiment Selection
    st.sidebar.header("Experiment Selection")

    experiments = find_experiments()

    if not experiments:
        st.warning("No experiments found. Start training to see data here.")
        st.info("Run: `python3 Main.py --config configs/base_config.yaml` (Tidal)")
        st.info("Or: `python3 Main.py --config configs/constant_base_config.yaml` (Constant)")
        return

    selected_exp = st.sidebar.selectbox(
        "Select Experiment",
        experiments,
        format_func=lambda x: x,
        key='experiment_selector'
    )

    # Paths
    exp_dir = os.path.join("experiments", selected_exp)

    # Detect model type
    model_type = get_model_type(exp_dir)
    model_icon = "🌊" if model_type == "tidal" else "📊"
    model_display_name = "Tidal Language Model" if model_type == "tidal" else "Constant Language Model (Baseline)"

    # Display model type prominently
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {model_icon} Model Type")
    st.sidebar.info(f"**{model_display_name}**")

    # Data Range Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Range")

    view_mode = st.sidebar.radio(
        "View Mode",
        ["Recent", "Historical (downsampled)"],
        help="Recent: Shows last N points. Historical: Downsampled view of all data."
    )

    if view_mode == "Recent":
        window_size = st.sidebar.select_slider(
            "Window Size",
            options=[1000, 5000, 10000, 25000, 50000],
            value=5000,
            help="Number of most recent data points to display"
        )
        max_points = 5000  # Default, not used in Recent mode
    else:
        max_points = st.sidebar.select_slider(
            "Max Points",
            options=[1000, 2500, 5000, 10000],
            value=5000,
            help="Maximum number of downsampled points to display"
        )
        window_size = 10000  # Default, not used in Historical mode

    # Refresh controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Refresh Controls")

    paused = st.sidebar.toggle("Pause auto-refresh", value=False, key="auto_refresh_paused")

    if st.sidebar.button("Refresh Now"):
        # Clear fingerprint to force data reload
        st.session_state.pop('_data_fingerprint', None)
        st.rerun()

    if paused:
        st.sidebar.caption("Auto-refresh paused. Use 'Refresh Now' for manual updates.")
    else:
        st.sidebar.caption("Auto-refreshes every 45s")

    # Training status display (static - renders once)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Training Status")
    status_file = os.path.join(exp_dir, "dashboard_metrics", "status.json")
    status = load_status(status_file)
    render_status_display(status)

    # Auto-refresh fragment - creates tabs and updates content every 45s
    auto_refresh_data_and_plots(exp_dir, model_type, view_mode, window_size, max_points)


if __name__ == "__main__":
    main()
