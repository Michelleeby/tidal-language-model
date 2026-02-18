"""
migrate_checkpoint.py

Migrates a 3-gate TransformerLM checkpoint to 1-gate (single modulation) format.

Each DynamicGate has a first Linear layer with weight shape [hidden_dim, 3].
This tool slices a single column (default: column 2, the stability gate) to
produce [hidden_dim, 1], matching the new GATE_DIM=1 architecture.

Usage:
    python -m plugins.tidal.migrate_checkpoint \
        --input old_checkpoint.pth \
        --output new_checkpoint.pth \
        [--slice-index 2]
"""

import argparse
import re
from collections import OrderedDict

import torch


# Pattern matches the first Linear weight in DynamicGate: net.0.weight
_GATE_FIRST_LINEAR_PATTERN = re.compile(
    r"transformer_blocks\.\d+\.(attn_gate|ffn_gate)\.net\.0\.weight"
)


def migrate_gate_dim(
    state_dict: dict,
    slice_index: int = 2,
) -> OrderedDict:
    """Migrate a 3-gate state_dict to 1-gate by slicing a single column.

    Args:
        state_dict: Original state_dict with [hidden, 3] gate weights.
        slice_index: Which column to keep (0=creativity, 1=focus, 2=stability).

    Returns:
        New state_dict with [hidden, 1] gate weights.
    """
    new_sd = OrderedDict()

    for key, tensor in state_dict.items():
        if _GATE_FIRST_LINEAR_PATTERN.match(key):
            # Only slice if input dim > 1 (skip already-migrated checkpoints)
            if tensor.shape[1] > 1:
                new_sd[key] = tensor[:, slice_index:slice_index + 1].clone()
            else:
                new_sd[key] = tensor.clone()
        else:
            new_sd[key] = tensor.clone()

    return new_sd


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a 3-gate checkpoint to 1-gate (single modulation)."
    )
    parser.add_argument("--input", required=True, help="Path to old checkpoint.")
    parser.add_argument("--output", required=True, help="Path to save migrated checkpoint.")
    parser.add_argument(
        "--slice-index", type=int, default=2,
        help="Gate column to keep (0=creativity, 1=focus, 2=stability). Default: 2.",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.input}")
    state_dict = torch.load(args.input, map_location="cpu")

    # Handle wrapped checkpoints (e.g. from Trainer)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        inner = state_dict["model_state_dict"]
        migrated = migrate_gate_dim(inner, slice_index=args.slice_index)
        state_dict["model_state_dict"] = migrated
    else:
        state_dict = migrate_gate_dim(state_dict, slice_index=args.slice_index)

    torch.save(state_dict, args.output)
    print(f"Migrated checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
