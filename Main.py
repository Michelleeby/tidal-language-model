import os
import sys
import argparse
import hashlib
import subprocess
import shutil
import multiprocessing
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from Trainer import Trainer
from Evaluator import Evaluator
from Utils import setup_logger

def get_git_commit_hash():
    """Gets the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Handle cases where it's not a git repo or git is not installed
        return "nogit"

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:10] # Use a truncated hash for brevity

def main():
    """
    Main entry point for running experiments.
    Handles experiment setup, ID generation, and orchestration of training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Run a Tidal Language Model experiment.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the experiment's YAML configuration file."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # 2. Generate Experiment ID
    commit_hash = get_git_commit_hash()
    config_hash = get_file_hash(args.config)
    experiment_id = f"commit_{commit_hash}-config_{config_hash}"
    
    # 3. Create Experiment Directory
    experiment_dir = os.path.join("experiments", experiment_id)
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup main logger for the orchestrator
    logger = setup_logger('MainOrchestrator', os.path.join(experiment_dir, 'main.log'), config)
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Configuration loaded from: {args.config}")

    # 4. Copy config for reproducibility
    shutil.copy(args.config, os.path.join(experiment_dir, 'config.yaml'))
    logger.info(f"Experiment directory created: {experiment_dir}")

    # 5. Run Training
    logger.info("="*20 + " INITIALIZING TRAINING " + "="*20)
    trainer = Trainer(config, experiment_dir)
    final_model_path = trainer.run()
    
    if final_model_path:
        logger.info(f"Training complete. Final model saved at: {final_model_path}")
        # 6. Run Evaluation
        logger.info("="*20 + " INITIALIZING EVALUATION " + "="*20)
        # You might want to add probe words to your config or pass them as args
        probe_words = config.get("PROBE_WORDS", ['water', 'love', 'think', 'ocean', 'freedom', 'grow'])
        evaluator = Evaluator(config, experiment_dir, final_model_path, probe_words)
        evaluator.run()
        logger.info("Evaluation complete.")
    else:
        logger.error("Training failed. Skipping evaluation.")

    logger.info(f"Experiment {experiment_id} finished.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
