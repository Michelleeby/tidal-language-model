"""
Generator.py

Inference / text generation script for the TransformerLM.
Supports standard generation and RL-controlled gated generation.
"""

import torch
import os
import sys
import argparse

# Add project root to sys.path for shared modules (MetricsLogger, experiment_utils)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ruamel.yaml import YAML

from TransformerLM import TransformerLM
from DataPipeline import get_tokenizer

# RL components (optional, for gating-controlled generation)
try:
    from GatingPolicyAgent import create_agent
    from GatingModulator import GatingModulator, RandomGatingPolicy, FixedGatingPolicy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


def run_generation(args):
    """Main function to load the model and run text generation."""
    print("--- TransformerLM Inference ---")

    # 1. Load Configuration
    print(f"Loading configuration from: {args.config}")
    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # 2. Load Tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = config.get("VOCAB_SIZE", tokenizer.vocab_size)

    # 3. Initialize Model
    print("Initializing TransformerLM...")
    model = TransformerLM(vocab_size=vocab_size, config=config)

    # 4. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    print(f"Loading model state from: {args.checkpoint}")
    device = "cuda" if torch.cuda.is_available() and config.get("DEVICE", "auto") != "cpu" else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    print("Model loaded successfully.")

    # 5. Prepare Prompt
    print(f"\nOriginal Prompt: '{args.prompt}'")
    encoded = tokenizer.encode(args.prompt)
    prompt_ids = torch.tensor(encoded, dtype=torch.long)
    if prompt_ids.nelement() == 0:
        print("Error: The prompt could not be tokenized.")
        return

    # 6. Run Generation
    print("\n--- Generating Text ---")

    if args.rl_agent and RL_AVAILABLE:
        print("Using RL-controlled gated generation...")

        rl_config_path = args.rl_config or "configs/rl_config.yaml"
        if os.path.exists(rl_config_path):
            with open(rl_config_path, "r") as f:
                rl_config = yaml.load(f)
            config.update(rl_config)

        modulator = GatingModulator(config)

        if args.rl_checkpoint:
            print(f"Loading RL agent from: {args.rl_checkpoint}")
            agent = create_agent(config, device)
            checkpoint = torch.load(args.rl_checkpoint, map_location=device)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            agent.eval()
            gating_policy = agent
        elif args.gating_mode == "random":
            print("Using random gating policy")
            gating_policy = RandomGatingPolicy(device)
        elif args.gating_mode == "fixed":
            print(f"Using fixed gating policy: creativity={args.creativity}, focus={args.focus}, stability={args.stability}")
            gating_policy = FixedGatingPolicy(
                creativity=args.creativity,
                focus=args.focus,
                stability=args.stability,
                device=device,
            )
        else:
            print("Using learned RL agent (default checkpoint)")
            checkpoint_path = os.path.join(os.path.dirname(args.checkpoint), "rl_checkpoint_final.pth")
            if os.path.exists(checkpoint_path):
                agent = create_agent(config, device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                agent.load_state_dict(checkpoint["agent_state_dict"])
                agent.eval()
                gating_policy = agent
            else:
                print(f"Warning: No RL checkpoint found at {checkpoint_path}, using fixed gating")
                gating_policy = FixedGatingPolicy(device=device)

        generated_ids, trajectory = model.generate_with_gating(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            gating_policy=gating_policy,
            modulator=modulator,
            base_temperature=args.temperature,
            top_k=args.top_k,
            return_trajectory=args.verbose,
        )

        if args.verbose and trajectory:
            print("\n--- Gating Trajectory ---")
            for i, action in enumerate(trajectory["actions"]):
                if i < 10 or i >= len(trajectory["actions"]) - 5:
                    print(f"  Step {i}: creativity={action[0]:.3f}, focus={action[1]:.3f}, stability={action[2]:.3f}")
                elif i == 10:
                    print("  ...")

    else:
        if args.rl_agent and not RL_AVAILABLE:
            print("Warning: RL components not available. Using standard generation.")

        generated_ids = model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

    generated_text = tokenizer.decode(generated_ids)
    print(f"Generated Text:\n{generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the TransformerLM.")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pth) file.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The starting text prompt for generation.")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature. Higher is more random.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering for sampling.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to the base configuration YAML file.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty for repeating tokens.")

    # RL gating control arguments
    parser.add_argument("--rl-agent", action="store_true",
                        help="Use RL-controlled gated generation.")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to trained RL agent checkpoint.")
    parser.add_argument("--rl-config", type=str, default=None,
                        help="Path to RL configuration file.")
    parser.add_argument("--gating-mode", type=str, default="learned",
                        choices=["learned", "random", "fixed"],
                        help="Gating control mode: learned (RL agent), random, or fixed.")
    parser.add_argument("--creativity", type=float, default=0.5,
                        help="Fixed creativity gate level (0-1). Used with --gating-mode fixed.")
    parser.add_argument("--focus", type=float, default=0.5,
                        help="Fixed focus gate level (0-1). Used with --gating-mode fixed.")
    parser.add_argument("--stability", type=float, default=0.5,
                        help="Fixed stability gate level (0-1). Used with --gating-mode fixed.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed gating trajectory during generation.")

    args = parser.parse_args()
    run_generation(args)
