import os
import sys
import argparse
import hashlib
import subprocess
import shutil
import torch
import time

from ruamel.yaml import YAML
yaml = YAML(typ="safe")

from Trainer import Trainer
from Evaluator import Evaluator
from Utils import setup_logger


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).strip().decode("utf-8")
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:10]


def main():
    """Main entry point for training and evaluating the TransformerLM."""
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Train and evaluate a TransformerLM.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing experiment directory to resume training from.")
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # 2. Resolve Experiment Directory
    if args.resume:
        experiment_dir = args.resume
        if not os.path.isdir(experiment_dir):
            print(f"Error: Resume directory not found: {experiment_dir}")
            sys.exit(1)
        experiment_id = os.path.basename(experiment_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        base_id = f"commit_{get_git_commit_hash()}-config_{get_file_hash(args.config)}"
        experiment_id = f"{timestamp}-{base_id}"
        experiment_dir = os.path.join("experiments", experiment_id)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # 3. Setup Logger
    logger = setup_logger("MainOrchestrator", os.path.join(experiment_dir, "main.log"), config)
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Dashboard: http://localhost:4400")
    logger.info(f"To view training progress, run: cd dashboard && ddev start && npm run dev")

    final_model_path = None
    try:
        # 4. Run Training
        logger.info("=" * 20 + " STARTING TRAINING " + "=" * 20)
        trainer = Trainer(config, experiment_dir)
        final_model_path = trainer.run()

        # 5. Run Evaluation (perplexity + sample generation)
        if final_model_path:
            logger.info("=" * 20 + " STARTING EVALUATION " + "=" * 20)
            evaluator = Evaluator(config, experiment_dir, final_model_path)
            evaluator.run()
        else:
            logger.warning("Training did not produce a final model. Skipping evaluation.")

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 20 + " KEYBOARD INTERRUPT " + "=" * 20)
        logger.warning("Experiment interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info(f"Experiment {experiment_id} finished.")


if __name__ == "__main__":
    main()
