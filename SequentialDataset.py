import os
import torch
import pickle

from torch.utils.data import Dataset

def load_or_create_dataset(corpus_path: str, config: dict) -> Dataset:
    """
    Loads a pre-processed sequential dataset from cache.
    """
    sequence_length = config.get("SEQUENCE_LENGTH", 50)
    cache_dir = config.get("CACHE_DIR", "cache")
    data_filename = f"{os.path.basename(corpus_path)}_seqlen{sequence_length}.pt"
    data_path = os.path.join(cache_dir, data_filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run Preprocess.py first.")

    print(f"Loading pre-processed data from {data_path}...")
    return SequentialDataset(data_path)

def load_vocab(config: dict) -> dict:
    """
    Loads the vocabulary from the cache.
    """
    cache_dir = config.get("CACHE_DIR", "cache")
    vocab_path = os.path.join(cache_dir, config.get("VOCAB_CACHE_PATH", "vocab.pkl"))
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Please run Preprocess.py first.")
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)

class SequentialDataset(Dataset):
    """
    Loads pre-processed sequences and serves them as (input, target) pairs.
    """
    def __init__(self, data_tensor_path: str):
        """
        Args:
            data_tensor_path (str): Path to the .pt file containing sequences.
        """
        self.sequences = torch.load(data_tensor_path)

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves a sequence and splits it into an input and a target.

        For a sequence [t1, t2, ..., tN]:
        - Input (x) is [t1, t2, ..., t(N-1)]
        - Target (y) is [t2, t3, ..., tN]
        
        This structure trains the model to predict the next word at each step.
        """
        sequence = self.sequences[idx]
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]
        return input_sequence, target_sequence