import os
import sys
import argparse
import hashlib
import subprocess
import shutil
import multiprocessing
import torch
import threading
import time

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

# Set matmul precision for better performance on Ampere GPUs and newer
torch.set_float32_matmul_precision('high')

from Trainer import Trainer
from Evaluator import Evaluator
from Utils import setup_logger

def get_git_commit_hash():
    """Gets the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:10]

def run_training(config, experiment_dir, final_model_path_ref, logger):
    """Target function for the training thread."""
    try:
        trainer = Trainer(config, experiment_dir)
        final_model_path = trainer.run()
        final_model_path_ref[0] = final_model_path
    except Exception as e:
        logger.error(f"TRAINING THREAD CRASHED: {e}", exc_info=True)
        final_model_path_ref[0] = None


def run_evaluation(config, experiment_dir, final_model_path_ref, logger):
    """Target function for the evaluation thread."""
    # Wait until the training thread is no longer in its initial "training" state.
    while final_model_path_ref[0] == "training":
        time.sleep(5) 

    # Check if training produced a valid model path.
    if final_model_path_ref[0]:
        logger.info("Evaluation thread started.")
        try:
            probe_words = config.get("PROBE_WORDS", ['water', 'love', 'think', 'ocean', 'freedom', 'grow'])
            evaluator = Evaluator(config, experiment_dir, final_model_path_ref[0], probe_words)
            evaluator.run()
            logger.info("Evaluation thread finished.")
        except Exception as e:
            logger.error(f"EVALUATION THREAD CRASHED: {e}", exc_info=True)
    else:
        logger.warning("Skipping evaluation due to training failure.")


def launch_dashboard(logger):
    """Target function for the dashboard thread."""
    try:
        logger.info("Attempting to launch Streamlit dashboard...")
        command = ["streamlit", "run", "Dashboard.py", "--server.runOnSave", "true"]
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        logger.error("DASHBOARD LAUNCH FAILED: `streamlit` command not found. Is it installed and in your system's PATH?")
    except subprocess.CalledProcessError as e:
        logger.error(f"DASHBOARD THREAD ERROR: Streamlit process failed with exit code {e.returncode}.")
        logger.error(f"Streamlit stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"DASHBOARD THREAD CRASHED: {e}", exc_info=True)


def main():
    """
    Main entry point for running experiments.
    Handles experiment setup and orchestrates concurrent training, evaluation, and dashboard launch.
    """
    parser = argparse.ArgumentParser(description="Run a Tidal Language Model experiment.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument('--no-dashboard', action='store_true', help="Do not launch the Streamlit dashboard.")
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # 2. Generate Experiment ID & Directory
    experiment_id = f"commit_{get_git_commit_hash()}-config_{get_file_hash(args.config)}"
    experiment_dir = os.path.join("experiments", experiment_id)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    shutil.copy(args.config, os.path.join(experiment_dir, 'config.yaml'))
    
    # 3. Setup Main Logger
    logger = setup_logger('MainOrchestrator', os.path.join(experiment_dir, 'main.log'), config)
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Experiment directory: {experiment_dir}")

    # 4. Setup and Launch Threads
    # A mutable list to act as a reference for passing the model path between threads.
    final_model_path_ref = ["training"] 

    # Create threads, passing the logger to each target function.
    training_thread = threading.Thread(target=run_training, args=(config, experiment_dir, final_model_path_ref, logger))
    evaluation_thread = threading.Thread(target=run_evaluation, args=(config, experiment_dir, final_model_path_ref, logger))
    
    dashboard_thread = None
    if not args.no_dashboard:
        dashboard_thread = threading.Thread(target=launch_dashboard, args=(logger,))
        dashboard_thread.daemon = True 

    # Start threads
    logger.info("="*20 + " LAUNCHING THREADS " + "="*20)
    training_thread.start()
    logger.info("🚀 Training thread started.")
    
    evaluation_thread.start()
    logger.info("⏳ Evaluation thread started (will wait for training to finish).")
    
    if dashboard_thread:
        dashboard_thread.start()
        logger.info("🌊 Dashboard thread started.")

    # Wait for Completion
    training_thread.join() 
    logger.info("✅ Training thread has finished.")
    
    evaluation_thread.join() 
    logger.info("✅ Evaluation thread has finished.")

    logger.info(f"Experiment {experiment_id} finished.")
    logger.info("Main orchestrator finished. Dashboard may still be running if launched.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
