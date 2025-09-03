import os
import torch
import pickle

from torch.utils.data import Dataset

def load_or_create_dataset(corpus_path: str, vocab: dict, config: dict) -> Dataset:
    """
    Loads a pre-processed dataset.
    """
    window_size = config.get("DEFAULT_WINDOW_SIZE", 2)
    cache_dir = config.get("CACHE_DIR", "cache")
    
    data_filename = f"{os.path.basename(corpus_path)}_ws{window_size}.pt"
    data_path = os.path.join(cache_dir, data_filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run preprocess.py first.")

    print(f"Loading pre-processed data from {data_path}...")
    return AssociativeDataset(data_path)

def load_vocab(config: dict) -> dict:
    """
    Loads the vocabulary from the cache.
    """
    cache_dir = config.get("CACHE_DIR", "cache")
    vocab_path = os.path.join(cache_dir, config.get("VOCAB_CACHE_PATH", "vocab.pkl"))
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Please run preprocess.py first.")
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)

class AssociativeDataset(Dataset):
    """
    Loads pre-processed (center, context) pairs from a file.
    """
    def __init__(self, data_tensor_path: str):
        self.data = torch.load(data_tensor_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, 0], self.data[idx, 1]
