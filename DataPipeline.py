"""
DataPipeline.py

Data loading and tokenization for training on TinyStories via HuggingFace.
Replaces the legacy Preprocess.py + SequentialDataset.py pipeline.

Usage:
    from DataPipeline import TinyStoriesDataset, get_tokenizer

    tokenizer = get_tokenizer()
    train_ds = TinyStoriesDataset("train", max_length=256, tokenizer=tokenizer)
    val_ds   = TinyStoriesDataset("validation", max_length=256, tokenizer=tokenizer)

    input_ids, target_ids = train_ds[0]  # both shape (max_length,)
"""

import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger("DataPipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def get_tokenizer():
    """Return configured GPT-2 BPE tokenizer (vocab size 50257)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # We handle our own chunking so suppress the "sequence longer than
    # model_max_length" warning that fires for long TinyStories entries.
    tokenizer.model_max_length = int(1e30)
    return tokenizer


def _load_dataset_prefer_cache(dataset_name, split):
    """Load HF dataset preferring local cache. Falls back to network on cache miss."""
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return load_dataset(dataset_name, split=split)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        logger.info("Local HF cache miss, downloading from HuggingFace...")
        return load_dataset(dataset_name, split=split)
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)


class TinyStoriesDataset(Dataset):
    """
    Loads roneneldan/TinyStories from HuggingFace, tokenizes with GPT-2 BPE,
    and serves fixed-length (input, target) pairs for language modelling.

    Each sample is a chunk of `max_length` contiguous token IDs.
    `input_ids = chunk[:-1]`, `target_ids = chunk[1:]` (standard causal LM shift).
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 256,
        tokenizer=None,
        dataset_name: str = "roneneldan/TinyStories",
    ):
        """
        Args:
            split: HuggingFace dataset split ("train" or "validation").
            max_length: Sequence length for each sample (model context length).
            tokenizer: GPT-2 tokenizer instance. Created if not provided.
            dataset_name: HuggingFace dataset identifier.
        """
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer or get_tokenizer()

        # chunk_length includes one extra token for the target shift
        chunk_length = max_length + 1

        # Try loading from local cache first
        cache_path = os.path.join(CACHE_DIR, f"{split}_ctx{max_length}.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached chunks from {cache_path}")
            self.chunks = torch.load(cache_path, weights_only=True)
            return

        # Cache miss — build chunks from HF dataset (one-time).
        # Try offline first (uses HF's local cache) to avoid slow Hub requests.
        logger.info(f"Building {split} chunk cache (one-time)...")
        logger.info("Loading dataset from HF cache...")
        raw = _load_dataset_prefer_cache(dataset_name, split)
        num_examples = len(raw)
        logger.info(f"Loaded {num_examples:,} examples. Tokenizing and flattening...")
        batch_size = 5000
        token_chunks: list[np.ndarray] = []
        total = 0
        for start in range(0, num_examples, batch_size):
            batch_texts = raw[start : start + batch_size]["text"]
            encoded = self.tokenizer(
                batch_texts,
                truncation=False,
                return_attention_mask=False,
            )
            for ids in encoded["input_ids"]:
                arr = np.array(ids, dtype=np.int64)
                token_chunks.append(arr)
                total += len(arr)
            if start % 50000 == 0:
                logger.info(f"  Tokenized {start + len(batch_texts):,}/{num_examples:,} examples ({total:,} tokens)")

        logger.info(f"Total tokens: {total:,}. Concatenating...")
        all_ids = np.empty(total, dtype=np.int64)
        offset = 0
        for arr in token_chunks:
            n = len(arr)
            all_ids[offset : offset + n] = arr
            offset += n

        # Drop remainder that doesn't fill a full chunk
        num_chunks = len(all_ids) // chunk_length
        all_ids = all_ids[: num_chunks * chunk_length]
        self.chunks = torch.from_numpy(all_ids).view(num_chunks, chunk_length)

        # Save to local cache — subsequent runs skip all of the above
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(self.chunks, cache_path)
        logger.info(f"Cached {num_chunks:,} chunks to {cache_path}")

    def __len__(self) -> int:
        return self.chunks.size(0)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        input_ids = chunk[:-1]    # (max_length,)
        target_ids = chunk[1:]    # (max_length,)
        return input_ids, target_ids
