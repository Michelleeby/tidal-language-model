"""
train_rl.py

Entry point for RL hormone controller training on a frozen ConstantLanguageModel.

Usage:
    # Train RL agent
    python3 train_rl.py \
        --config configs/constant_base_config.yaml \
        --rl-config configs/rl_config.yaml \
        --checkpoint experiments/<experiment_id>/constant-baseline_v1.0.0.pth

    # Resume from RL checkpoint
    python3 train_rl.py \
        --config configs/constant_base_config.yaml \
        --rl-config configs/rl_config.yaml \
        --checkpoint experiments/<experiment_id>/constant-baseline_v1.0.0.pth \
        --resume experiments/<rl_experiment_id>/rl_checkpoint_iter_100.pth

    # Run ablation study after training
    python3 train_rl.py \
        --config configs/constant_base_config.yaml \
        --rl-config configs/rl_config.yaml \
        --checkpoint experiments/<experiment_id>/constant-baseline_v1.0.0.pth \
        --ablation
"""

import os
import sys
import argparse
import hashlib
import shutil
import subprocess
import time
import pickle

import torch
from ruamel.yaml import YAML

from ConstantLanguageModel import ConstantLanguageModel
from SequentialDataset import load_vocab, load_or_create_dataset
from HormoneRLAgent import create_agent
from HormoneEnvironment import HormoneEnvironment
from HormoneModulator import HormoneModulator
from RewardComputer import RewardComputer
from RLTrainer import PPOTrainer, run_ablation_study

yaml = YAML(typ='safe')


def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:10]


def load_model(config, checkpoint_path, device):
    """Load a frozen ConstantLanguageModel from checkpoint."""
    vocab = load_vocab(config)
    vocab_size = len(vocab)

    model = ConstantLanguageModel(vocab_size=vocab_size, config=config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, vocab


def extract_prompt_tokens(config, min_length=3, max_length=10):
    """Extract prompt token sequences from the training dataset."""
    corpus_path = config.get("FOUNDATIONAL_CORPUS_PATH", "corpus/foundational.txt")
    dataset = load_or_create_dataset(corpus_path, config)

    prompts = []
    for i in range(len(dataset)):
        seq = dataset[i]
        # seq is (input, target) tuple from SequentialDataset
        if isinstance(seq, (tuple, list)):
            tokens = seq[0]
        else:
            tokens = seq

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if len(tokens) >= min_length:
            prompts.append(tokens[:max_length])

    print(f"Extracted {len(prompts)} prompts from training data")
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL hormone controller on a frozen ConstantLanguageModel."
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to the base model YAML config (e.g. configs/constant_base_config.yaml)"
    )
    parser.add_argument(
        '--rl-config', type=str, required=True,
        help="Path to the RL YAML config (e.g. configs/rl_config.yaml)"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help="Path to trained ConstantLanguageModel checkpoint (.pth)"
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Path to RL checkpoint to resume training from"
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help="Run ablation study (learned vs random vs fixed vs no-hormone)"
    )
    parser.add_argument(
        '--timesteps', type=int, default=None,
        help="Override RL_TOTAL_TIMESTEPS from config"
    )
    parser.add_argument(
        '--experiment-dir', type=str, default=None,
        help="Existing RL experiment directory (for --ablation to find trained checkpoint)"
    )
    args = parser.parse_args()

    # Validate paths
    for path, name in [(args.config, "config"), (args.rl_config, "rl-config"), (args.checkpoint, "checkpoint")]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            sys.exit(1)

    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    with open(args.rl_config, 'r') as f:
        rl_config = yaml.load(f)

    # Merge RL config into base config (RL keys override)
    merged_config = {**config, **rl_config}

    if args.timesteps is not None:
        merged_config["RL_TOTAL_TIMESTEPS"] = args.timesteps

    # Device setup
    device_str = merged_config.get("DEVICE", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Determine experiment directory
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        if not os.path.isdir(experiment_dir):
            print(f"Error: experiment directory not found at {experiment_dir}")
            sys.exit(1)
        experiment_id = os.path.basename(experiment_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        base_id = f"commit_{get_git_commit_hash()}-rl_{get_file_hash(args.rl_config)}"
        experiment_id = f"{timestamp}-{base_id}"
        experiment_dir = os.path.join("experiments", experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(experiment_dir, 'base_config.yaml'))
        shutil.copy(args.rl_config, os.path.join(experiment_dir, 'rl_config.yaml'))

    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {experiment_dir}")

    # Load frozen language model
    print(f"\nLoading model from {args.checkpoint}...")
    model, vocab = load_model(config, args.checkpoint, device)
    idx_to_word = {v: k for k, v in vocab.items()}
    print(f"Model loaded (vocab_size={model.vocab_size}, device={device})")

    # Extract prompts from training data
    print("\nExtracting prompts from training corpus...")
    prompt_tokens = extract_prompt_tokens(
        config,
        min_length=merged_config.get("RL_PROMPT_MIN_LENGTH", 3),
        max_length=merged_config.get("RL_PROMPT_MAX_LENGTH", 10)
    )

    if len(prompt_tokens) == 0:
        print("Error: No valid prompts extracted from training data.")
        sys.exit(1)

    # Create RL components
    print("\nInitializing RL components...")
    modulator = HormoneModulator(merged_config)
    reward_computer = RewardComputer(merged_config, model.vocab_size)

    # Build n-gram statistics for coherence reward
    print("Building n-gram statistics for reward computation...")
    reward_computer.build_ngram_statistics(prompt_tokens)

    # Create environment
    env = HormoneEnvironment(
        model=model,
        modulator=modulator,
        reward_computer=reward_computer,
        prompt_tokens=prompt_tokens,
        config=merged_config,
        device=device
    )

    # Create RL agent
    agent = create_agent(merged_config, device)
    agent_type = merged_config.get("RL_AGENT_TYPE", "beta")
    param_count = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {agent_type} ({param_count:,} parameters)")

    if args.ablation:
        # Run ablation study
        print("\n" + "=" * 60)
        print("RUNNING ABLATION STUDY")
        print("=" * 60)

        results = run_ablation_study(
            model=model,
            prompt_tokens=prompt_tokens,
            config=merged_config,
            experiment_dir=experiment_dir,
            device=device
        )

        print("\n" + "=" * 60)
        print("ABLATION RESULTS")
        print("=" * 60)
        for policy_name, metrics in results.items():
            print(f"\n{policy_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        return

    # Create PPO trainer
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        config=merged_config,
        experiment_dir=experiment_dir,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: RL checkpoint not found at {args.resume}")
            sys.exit(1)
        print(f"\nResuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    # Train
    total_timesteps = merged_config.get("RL_TOTAL_TIMESTEPS", 100000)
    print(f"\n{'=' * 60}")
    print(f"STARTING RL TRAINING")
    print(f"{'=' * 60}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Rollout steps:   {merged_config.get('RL_ROLLOUT_STEPS', 128)}")
    print(f"  PPO epochs:      {merged_config.get('RL_NUM_EPOCHS', 4)}")
    print(f"  Batch size:      {merged_config.get('RL_BATCH_SIZE', 32)}")
    print(f"  Learning rate:   {merged_config.get('RL_LEARNING_RATE', 3e-4)}")
    print(f"{'=' * 60}\n")

    try:
        history = trainer.train(total_timesteps)
        print(f"\nTraining complete. Final checkpoint saved to {experiment_dir}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("rl_checkpoint_interrupted.pth")
        print(f"Checkpoint saved to {experiment_dir}/rl_checkpoint_interrupted.pth")

    # Run ablation study after training
    print(f"\n{'=' * 60}")
    print("POST-TRAINING ABLATION STUDY")
    print(f"{'=' * 60}")

    results = run_ablation_study(
        model=model,
        prompt_tokens=prompt_tokens,
        config=merged_config,
        experiment_dir=experiment_dir,
        device=device
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
