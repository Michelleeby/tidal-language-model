import logging
import os
import matplotlib.pyplot as plt

def setup_logger(name: str, log_file: str, config: dict):
    """
    Sets up a logger with a file handler and a console handler.

    Args:
        name (str): The name for the logger.
        log_file (str): The file to which logs should be written.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Create the main log directory if it doesn't exist
    os.makedirs(config["LOG_DIRECTORY"], exist_ok=True)

    # Create file handler for detailed DEBUG logs
    log_path = os.path.join(config["LOG_DIRECTORY"], log_file)

    logger = logging.getLogger(name)
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # This check prevents adding handlers multiple times if the function is called again
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Capture all levels of messages

        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Create console handler for general INFO updates
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

def plot_semantic_space(ax, positions, forces, epoch_num, batch_num, title_suffix=""):
    """
    Renders a 2D scatter plot of word positions and their force vectors,
    only including points within the defined viewport.
    """
    # Define the viewport limits
    x_lim = (-5, 5)
    y_lim = (-5, 5)

    # This checks which positions are within our x and y limits.
    visible_mask = (positions[:, 0] >= x_lim[0]) & (positions[:, 0] <= x_lim[1]) & \
                   (positions[:, 1] >= y_lim[0]) & (positions[:, 1] <= y_lim[1])
    
    # Apply the mask to get only the data for visible points
    visible_positions = positions[visible_mask]
    visible_forces = forces[visible_mask]

    # Move data to CPU and convert to NumPy for plotting
    visible_positions_np = visible_positions.cpu().numpy()
    visible_forces_np = visible_forces.cpu().numpy()
    
    ax.clear()
    
    # Plot word positions
    ax.scatter(visible_positions_np[:, 0], visible_positions_np[:, 1], alpha=0.6, s=20, label="Concepts")
    
    # Plot force vectors using a quiver plot
    ax.quiver(visible_positions_np[:, 0], visible_positions_np[:, 1], 
              visible_forces_np[:, 0], visible_forces_np[:, 1], 
              color='r', alpha=0.4, width=0.003, scale=1.0, scale_units='xy')
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Semantic Space - Epoch {epoch_num}, Batch {batch_num} {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Set fixed limits to prevent the plot from resizing wildly
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Add a legend that also shows how many concepts are visible
    num_total = positions.shape[0]
    num_visible = visible_positions.shape[0]
    ax.legend([f"Concepts ({num_visible}/{num_total} visible)"])

    plt.pause(0.01)