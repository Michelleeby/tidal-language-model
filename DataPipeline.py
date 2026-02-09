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

import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset


def get_tokenizer() -> GPT2TokenizerFast:
    """Return configured GPT-2 BPE tokenizer (vocab size 50257)."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
        tokenizer: GPT2TokenizerFast = None,
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

        # Load and tokenize
        raw = load_dataset(dataset_name, split=split)
        tokenized = raw.map(
            lambda batch: self.tokenizer(
                batch["text"],
                truncation=False,
                return_attention_mask=False,
            ),
            batched=True,
            remove_columns=raw.column_names,
            desc=f"Tokenizing {split}",
        )

        # Flatten all tokens into a single 1-D list, then chunk
        all_ids = []
        for example in tokenized:
            all_ids.extend(example["input_ids"])

        # Drop remainder that doesn't fill a full chunk
        num_chunks = len(all_ids) // chunk_length
        all_ids = all_ids[: num_chunks * chunk_length]
        self.chunks = torch.tensor(all_ids, dtype=torch.long).view(num_chunks, chunk_length)

    def __len__(self) -> int:
        return self.chunks.size(0)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        input_ids = chunk[:-1]    # (max_length,)
        target_ids = chunk[1:]    # (max_length,)
        return input_ids, target_ids
