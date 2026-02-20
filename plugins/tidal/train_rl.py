"""
train_rl.py

Entry point for RL gating controller training on a frozen TransformerLM.

Each RL run creates its own independent experiment directory with a
metadata.json that links back to the source LM experiment and checkpoint.
This allows multiple RL runs on the same base model without overwriting.

Usage:
    # Train RL agent (creates a new experiment directory)
    python3 train_rl.py \
        --config configs/base_config.yaml \
        --rl-config configs/rl_config.yaml \
        --checkpoint experiments/<lm_experiment_id>/transformer-lm_v1.0.0.pth

    # Resume from RL checkpoint
    python3 train_rl.py \
        --config configs/base_config.yaml \
        --rl-config configs/rl_config.yaml \
        --checkpoint experiments/<lm_experiment_id>/transformer-lm_v1.0.0.pth \
        --resume experiments/<rl_experiment_id>/rl_checkpoint_iter_100.pth

    # Run ablation study on an existing RL experiment
    python3 train_rl.py \
        --config configs/base_config.yaml \
        --rl-config configs/rl_config.yaml \
        --checkpoint experiments/<lm_experiment_id>/transformer-lm_v1.0.0.pth \
        --experiment-dir experiments/<rl_experiment_id> \
        --ablation
"""

import os
import sys
import argparse
import shutil
import time

import torch
from ruamel.yaml import YAML

from plugins.tidal.TransformerLM import TransformerLM
from plugins.tidal.DataPipeline import TinyStoriesDataset, get_tokenizer
from plugins.tidal.GatingPolicyAgent import create_agent
from plugins.tidal.GatingEnvironment import GatingEnvironment
from plugins.tidal.GatingModulator import GatingModulator
from plugins.tidal.RewardComputer import RewardComputer
from plugins.tidal.RLTrainer import PPOTrainer, run_ablation_study
from MetricsLogger import MetricsLogger
from experiment_utils import (
    get_git_commit_hash,
    get_file_hash,
    resolve_device,
    report_experiment_id_to_job,
    create_experiment_metadata,
    write_experiment_metadata,
)

yaml = YAML(typ="safe")


def load_model(config, checkpoint_path, device):
    """Load a frozen TransformerLM from checkpoint."""
    vocab_size = config.get("VOCAB_SIZE", 50257)

    model = TransformerLM(vocab_size=vocab_size, config=config)
    data = torch.load(checkpoint_path, map_location=device)
    if isinstance(data, dict) and "model_state_dict" in data:
        model.load_state_dict(data["model_state_dict"])
    else:
        model.load_state_dict(data)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def extract_prompt_tokens(config, min_length=3, max_length=10):
    """Extract prompt token sequences from TinyStories validation set."""
    tokenizer = get_tokenizer()
    ds = TinyStoriesDataset(
        split="validation",
        max_length=config.get("MAX_CONTEXT_LENGTH", 256),
        tokenizer=tokenizer,
    )

    prompts = []
    for i in range(min(len(ds), 5000)):
        input_ids, _ = ds[i]
        tokens = input_ids.tolist()
        if len(tokens) >= min_length:
            prompts.append(tokens[:max_length])

    print(f"Extracted {len(prompts)} prompts from TinyStories validation")
    return prompts


def create_rl_experiment_dir(
    checkpoint_path: str,
    rl_config_path: str,
    base_config_path: str,
    experiments_root: str = "experiments",
) -> tuple[str, str]:
    """Create a new independent experiment directory for RL training.

    Always creates a fresh directory so multiple RL runs on the same
    LM checkpoint don't overwrite each other.

    Returns:
        (experiment_dir, experiment_id)
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    base_id = f"commit_{get_git_commit_hash()}-rl_{get_file_hash(rl_config_path)}"
    experiment_id = f"{timestamp}-{base_id}"
    experiment_dir = os.path.join(experiments_root, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # Derive source experiment from checkpoint path
    checkpoint_abs = os.path.abspath(checkpoint_path)
    experiments_abs = os.path.abspath(experiments_root)
    checkpoint_parent = os.path.dirname(checkpoint_abs)

    if checkpoint_parent.startswith(experiments_abs) and checkpoint_parent != experiments_abs:
        source_experiment_id = os.path.basename(checkpoint_parent)
    else:
        source_experiment_id = None

    # Write metadata linking to source
    metadata = create_experiment_metadata(
        "rl",
        source_experiment_id=source_experiment_id,
        source_checkpoint=checkpoint_path,
    )
    write_experiment_metadata(experiment_dir, metadata)

    # Copy both configs into the new experiment directory
    shutil.copy(base_config_path, os.path.join(experiment_dir, "base_config.yaml"))
    shutil.copy(rl_config_path, os.path.join(experiment_dir, "rl_config.yaml"))

    return experiment_dir, experiment_id


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL gating controller on a frozen TransformerLM."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the base model YAML config (e.g. configs/base_config.yaml)",
    )
    parser.add_argument(
        "--rl-config", type=str, required=True,
        help="Path to the RL YAML config (e.g. configs/rl_config.yaml)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained TransformerLM checkpoint (.pth)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to RL checkpoint to resume training from",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study (learned vs random vs fixed vs neutral gating)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override RL_TOTAL_TIMESTEPS from config",
    )
    parser.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Existing RL experiment directory (for --ablation to find trained checkpoint)",
    )
    args = parser.parse_args()

    for path, name in [
        (args.config, "config"),
        (args.rl_config, "rl-config"),
        (args.checkpoint, "checkpoint"),
    ]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.load(f)
    with open(args.rl_config, "r") as f:
        rl_config = yaml.load(f)

    merged_config = {**config, **rl_config}

    if args.timesteps is not None:
        merged_config["RL_TOTAL_TIMESTEPS"] = args.timesteps

    device = resolve_device(merged_config)
    print(f"Using device: {device}")

    if args.experiment_dir:
        # Explicit directory (e.g. for --ablation on an existing RL experiment)
        experiment_dir = args.experiment_dir
        if not os.path.isdir(experiment_dir):
            print(f"Error: experiment directory not found at {experiment_dir}")
            sys.exit(1)
        experiment_id = os.path.basename(experiment_dir)
    else:
        # Always create a new independent experiment directory
        experiment_dir, experiment_id = create_rl_experiment_dir(
            checkpoint_path=args.checkpoint,
            rl_config_path=args.rl_config,
            base_config_path=args.config,
        )
        report_experiment_id_to_job(experiment_id)

    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {experiment_dir}")

    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(config, args.checkpoint, device)
    print(f"Model loaded (vocab_size={model.vocab_size}, device={device})")

    print("\nExtracting prompts from TinyStories validation...")
    prompt_tokens = extract_prompt_tokens(
        merged_config,
        min_length=merged_config.get("RL_PROMPT_MIN_LENGTH", 3),
        max_length=merged_config.get("RL_PROMPT_MAX_LENGTH", 10),
    )

    if len(prompt_tokens) == 0:
        print("Error: No valid prompts extracted.")
        sys.exit(1)

    print("\nInitializing RL components...")
    modulator = GatingModulator(merged_config)
    reward_computer = RewardComputer(merged_config, model.vocab_size)

    print("Building n-gram statistics for reward computation...")
    reward_computer.build_ngram_statistics(prompt_tokens)

    env = GatingEnvironment(
        model=model,
        modulator=modulator,
        reward_computer=reward_computer,
        prompt_tokens=prompt_tokens,
        config=merged_config,
        device=device,
    )

    agent = create_agent(merged_config, device)
    agent_type = merged_config.get("RL_AGENT_TYPE", "beta")
    param_count = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {agent_type} ({param_count:,} parameters)")

    if args.ablation:
        print("\n" + "=" * 60)
        print("RUNNING ABLATION STUDY")
        print("=" * 60)

        results = run_ablation_study(
            model=model,
            prompt_tokens=prompt_tokens,
            config=merged_config,
            experiment_dir=experiment_dir,
            device=device,
        )

        print("\n" + "=" * 60)
        print("ABLATION RESULTS")
        print("=" * 60)
        for policy_name, metrics in results.items():
            print(f"\n{policy_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        return

    metrics_logger = MetricsLogger(experiment_dir, reset_metrics=True)

    trainer = PPOTrainer(
        agent=agent,
        env=env,
        config=merged_config,
        experiment_dir=experiment_dir,
        device=device,
        metrics_logger=metrics_logger,
    )

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: RL checkpoint not found at {args.resume}")
            sys.exit(1)
        print(f"\nResuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    total_timesteps = merged_config.get("RL_TOTAL_TIMESTEPS", 100000)
    print(f"\n{'=' * 60}")
    print("STARTING RL TRAINING")
    print(f"{'=' * 60}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Rollout steps:   {merged_config.get('RL_ROLLOUT_STEPS', 128)}")
    print(f"  PPO epochs:      {merged_config.get('RL_NUM_EPOCHS', 4)}")
    print(f"  Batch size:      {merged_config.get('RL_BATCH_SIZE', 32)}")
    print(f"  Learning rate:   {merged_config.get('RL_LEARNING_RATE', 3e-4)}")
    print(f"{'=' * 60}\n")

    try:
        history = trainer.train(total_timesteps)
        metrics_logger.finalize()
        print(f"\nTraining complete. Final checkpoint saved to {experiment_dir}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("rl_checkpoint_interrupted.pth")
        print(f"Checkpoint saved to {experiment_dir}/rl_checkpoint_interrupted.pth")

    print(f"\n{'=' * 60}")
    print("POST-TRAINING ABLATION STUDY")
    print(f"{'=' * 60}")

    results = run_ablation_study(
        model=model,
        prompt_tokens=prompt_tokens,
        config=merged_config,
        experiment_dir=experiment_dir,
        device=device,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
