import torch
import spacy
from collections import Counter
import pickle
import os
from ruamel.yaml import YAML

nlp = None

def initialize_nlp(config: dict):
    """
    Loads the spaCy model using the provided configuration if it hasn't been loaded yet.
    """
    global nlp
    if nlp is None:
        model_name = config.get("NLP_MODEL", "en_core_web_sm")
        # For nlp.pipe on lines, we don't need the parser, which speeds things up.
        disabled_components = config.get("NLP_DISABLE", ["parser", "ner"])
            
        print(f"Loading spaCy model '{model_name}'...")
        nlp = spacy.load(model_name, disable=disabled_components)


def build_vocab(corpus_path: str, config: dict) -> dict:
    """
    Builds a vocabulary from a corpus file using nlp.pipe for efficient, batched processing.
    """
    cache_dir = config.get("CACHE_DIR", "cache")
    vocab_cache_file = config.get("VOCAB_CACHE_PATH", "vocab.pkl")
    cache_path = os.path.join(cache_dir, vocab_cache_file)
    
    if os.path.exists(cache_path):
        print(f"Loading cached vocabulary from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Cache not found. Building vocabulary from foundational corpus: {corpus_path}...")
    min_freq = config.get("VOCAB_MIN_FREQUENCY", 5)
    encoding = config.get("CORPUS_ENCODING", "utf-8")
    
    if nlp is None:
        raise RuntimeError("spaCy model is not initialized. Call initialize_nlp() first.")

    word_counts = Counter()
    print("Tokenizing foundational corpus...")
    
    # Use nlp.pipe to process the file as a stream of lines.
    # This is highly memory-efficient and leverages multiple cores.
    with open(corpus_path, 'r', encoding=encoding) as f:
        # We can tune batch_size based on memory and CPU cores.
        for doc in nlp.pipe(f, batch_size=1000):
            tokens = [token.text.lower() for token in doc if not token.is_space]
            word_counts.update(tokens)
    
    vocab = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[BOS]': 2,
        '[EOS]': 3
    }
    
    current_idx = len(vocab)
    frequent_words = [word for word, count in word_counts.items() if count >= min_freq]
    for word in frequent_words:
        if word not in vocab:
            vocab[word] = current_idx
            current_idx += 1

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary of {len(vocab)} words (including special tokens) cached to {cache_path}")

    return vocab


def preprocess_corpora(config_path: str):
    """
    Builds a vocabulary, then processes each corpus using nlp.pipe to create
    and cache training sequences efficiently.
    """
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    corpora_paths = {
        "foundational": config["FOUNDATIONAL_CORPUS_PATH"],
        "high": config["HIGH_TIDE_CORPUS_PATH"],
        "low": config["LOW_TIDE_CORPUS_PATH"],
        "storm": config["STORM_TIDE_CORPUS_PATH"]
    }

    sequence_length = config.get("SEQUENCE_LENGTH", 50)
    cache_dir = config.get("CACHE_DIR", "cache")
    encoding = config.get("CORPUS_ENCODING", "utf-8")

    initialize_nlp(config)

    foundational_corpus_path = config["FOUNDATIONAL_CORPUS_PATH"]
    vocab = build_vocab(foundational_corpus_path, config)
    print(f"--- Vocabulary loaded with {len(vocab)} words ---")

    print("\n--- Starting Training Pair Generation ---")
    for name, corpus_path in corpora_paths.items():
        print(f"\n>>> Processing corpus: '{name}'")

        data_path = os.path.join(cache_dir, f"{os.path.basename(corpus_path)}_seqlen{sequence_length}.pt")

        if os.path.exists(data_path):
            print(f"Pre-processed data already exists at {data_path}. Skipping.")
            continue
        
        print(f"Reading and tokenizing {corpus_path} with nlp.pipe...")
        all_token_ids = []
        unk_token_id = vocab['[UNK]']
        bos_token_id = vocab['[BOS]']
        eos_token_id = vocab['[EOS]']
        
        with open(corpus_path, 'r', encoding=encoding) as f:
            for doc in nlp.pipe(f, batch_size=1000):
                # NOTE: This approach treats each line as a "sentence". This is a robust
                # trade-off for performance on large, line-broken texts like from Project Gutenberg.
                # It avoids the memory overhead of finding true sentence boundaries across the whole file.
                if doc.has_annotation("SENT_START"):
                    for sent in doc.sents:
                        line_ids = [bos_token_id]
                        line_ids.extend([vocab.get(token.text.lower(), unk_token_id) for token in sent if not token.is_space])
                        line_ids.append(eos_token_id)
                        all_token_ids.extend(line_ids)
                else:
                    line_ids = [bos_token_id]
                    line_ids.extend([vocab.get(token.text.lower(), unk_token_id) for token in doc if not token.is_space])
                    line_ids.append(eos_token_id)
                    all_token_ids.extend(line_ids)
                    
        print(f"Corpus processed into {len(all_token_ids)} tokens (including BOS/EOS).")

        print("Generating training sequences...")
        sequences = []
        if len(all_token_ids) >= sequence_length:
            for i in range(len(all_token_ids) - sequence_length + 1):
                chunk = all_token_ids[i:i + sequence_length]
                sequences.append(chunk)

        if not sequences:
            print(f"Warning: No sequences were generated for this corpus (not enough tokens for sequence length {sequence_length}).")
            continue

        data_tensor = torch.tensor(sequences, dtype=torch.long)
        torch.save(data_tensor, data_path)
        print(f"Saved {len(data_tensor)} sequences to {data_path}")

    print("\n--- Pre-processing complete! ---")

if __name__ == '__main__':
    config_path = os.path.join('configs', 'base_config.yaml')
    preprocess_corpora(config_path)