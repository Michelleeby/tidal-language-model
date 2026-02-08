import logging
import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def setup_logger(name: str, log_file: str, config: dict):
    """
    Sets up a logger with a file handler and an optional console handler.

    Args:
        name (str): The name for the logger.
        log_file (str): The file to which logs should be written.
        config (dict): Configuration dictionary with settings like LOG_DIRECTORY.

    Returns:
        logging.Logger: The configured logger instance.
    """
    log_path = os.path.join(config.get("LOG_DIRECTORY", "logs"), log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    
    # This check prevents adding handlers multiple times if the function is called again.
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Capture all levels of messages.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        def _create_and_add_handler(handler_class, level, **kwargs):
            """Helper to create, configure, and add a handler to the logger."""
            handler = handler_class(**kwargs)
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Always add the file handler.
        _create_and_add_handler(
            logging.FileHandler,
            level=logging.DEBUG,
            filename=log_path,
            mode='w',
            encoding='utf-8'
        )

        # Conditionally add the console stream handler.
        if config.get("ENABLE_CONSOLE_LOGGING", True):
            _create_and_add_handler(
                logging.StreamHandler,
                level=logging.INFO,
                stream=sys.stdout
            )
            
    return logger

def plot_semantic_space(fig, ax, viz_data, token_ids, vocab, probe_words, epoch_num, batch_num, title_suffix="", config=None):
    """
    Renders an enhanced 2D scatter plot that clusters concepts in 8D semantic space
    and labels probe words and the heaviest concepts in the current view.
    """
    x_lim = (-5, 5)
    y_lim = (-5, 5)

    positions_2d_cpu = viz_data['positions_2d'].cpu()
    positions_8d_cpu = viz_data['positions_8d'].cpu()
    masses_cpu = viz_data['masses'].cpu().squeeze()

    num_clusters = 8
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(positions_8d_cpu)
    
    ax.clear()
    
    scatter = ax.scatter(
        positions_2d_cpu[:, 0], positions_2d_cpu[:, 1],
        s=(masses_cpu * 40).numpy(),     # Size by mass
        c=cluster_labels,               # Color by cluster ID
        cmap='tab10',                   # A good colormap for distinct categories
        alpha=0.8
    )

    axis_mapping = {v: k for k, v in config["SEMANTIC_AXIS_MAPPING"].items()}
    cluster_centers_8d = kmeans.cluster_centers_
    legend_labels = []
    for i in range(num_clusters):
        center = cluster_centers_8d[i]
        # Find the two axes with the largest absolute values.
        top_two_indices = np.argsort(np.abs(center))[-2:]
        
        # Create labels like "High G_AXIS / Low V_AXIS" to pair with the Semantic Space concepts.
        label_parts = []
        for axis_idx in top_two_indices:
            axis_name = axis_mapping.get(axis_idx, f"AXIS_{axis_idx}")
            value = center[axis_idx]
            prefix = "High" if value > 0 else "Low"
            label_parts.append(f"{prefix} {axis_name.replace('_AXIS','')}")
        legend_labels.append(" / ".join(label_parts))

    # Create a legend that maps colors to the new descriptive labels.
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                                  markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                       for i in range(num_clusters)]
    ax.legend(handles=legend_elements, title="Cluster Characteristics", fontsize='small')

    idx_to_word = {v: k for k, v in vocab.items()}
    token_ids_list = token_ids.cpu().tolist()
    
    # Create a set of already labeled words to avoid duplicates.
    labeled_indices = set()

    for word in probe_words:
        if vocab.get(word) in token_ids_list:
            try:
                idx = token_ids_list.index(vocab[word])
                pos = positions_2d_cpu[idx]
                if x_lim[0] <= pos[0] <= x_lim[1] and y_lim[0] <= pos[1] <= y_lim[1]:
                    ax.text(pos[0], pos[1], word, fontsize=9, ha='right', color='black', fontweight='bold')
                    labeled_indices.add(idx)
            except ValueError:
                continue

    # Label the Top 5 Heaviest (Largest Mass) Words in the current view
    # Find indices of all points within the viewport.
    visible_mask = (positions_2d_cpu[:, 0] >= x_lim[0]) & (positions_2d_cpu[:, 0] <= x_lim[1]) & \
                   (positions_2d_cpu[:, 1] >= y_lim[0]) & (positions_2d_cpu[:, 1] <= y_lim[1])
    visible_indices = torch.where(visible_mask)[0]
    
    if visible_indices.numel() > 0:
        visible_masses = masses_cpu[visible_indices]
        # Get the top 5 heaviest masses among visible points.
        num_to_label = min(5, len(visible_masses))
        top_mass_indices_local = torch.topk(visible_masses, num_to_label).indices
        # Convert back to original batch indices.
        top_mass_indices_global = visible_indices[top_mass_indices_local]

        for idx in top_mass_indices_global:
            if idx.item() not in labeled_indices: # Only label if not already a probe word.
                pos = positions_2d_cpu[idx]
                word = idx_to_word.get(token_ids_list[idx.item()], "N/A")
                ax.text(pos[0], pos[1], word, fontsize=8, ha='left', color='dimgray')

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Semantic Space (Colored by 8D Cluster) - Epoch {epoch_num}, Batch {batch_num} {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return kmeans

def format_cluster_analysis_text(cluster_centers, config):
    """Generates a markdown-formatted string analyzing cluster centroids."""
    md_string = "# Semantic Space 8D Cluster Centroid Analysis\n"
    axis_mapping = {v: k for k, v in config["SEMANTIC_AXIS_MAPPING"].items()}
    planes = {
        "Core Conceptual": config["SEMANTIC_SPACE_SLICES"]["core_conceptual"],
        "Affective": config["SEMANTIC_SPACE_SLICES"]["affective"],
        "Interoceptive": config["SEMANTIC_SPACE_SLICES"]["interoceptive"],
        "Structural": config["SEMANTIC_SPACE_SLICES"]["structural"],
    }

    for i, center in enumerate(cluster_centers):
        md_string += f"## Cluster {i}\n"
        for plane_name, (start, end) in planes.items():
            md_string += f"**{plane_name} Plane:**\n"
            for axis_idx in range(start, end):
                axis_name = axis_mapping.get(axis_idx, f"AXIS_{axis_idx}").replace('_AXIS', '')
                value = center[axis_idx]
                md_string += f"- **{axis_name}:** {value:.2f}\n"
        md_string += "\n---\n"
    return md_string