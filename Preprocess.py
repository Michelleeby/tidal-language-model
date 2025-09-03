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
        disabled_components = config.get("NLP_DISABLE", ["parser", "ner"])
        print(f"Loading spaCy model '{model_name}'...")
        nlp = spacy.load(model_name, disable=disabled_components)

def build_vocab(corpus_path: str, config: dict) -> dict:
    """
    Builds a vocabulary from a corpus file using lemmatization.
    It uses a default cache path from the config to prevent rebuilding.
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
    
    with open(corpus_path, 'r', encoding=encoding) as f:
        text = f.read()

    # Ensure the global nlp model is initialized
    if nlp is None:
        raise RuntimeError("spaCy model is not initialized. Call initialize_nlp() first.")

    nlp.max_length = len(text) + config.get("NLP_PADDING_SIZE", 100)
    doc = nlp(text)

    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    word_counts = Counter(lemmas)

    frequent_words = [word for word, count in word_counts.items() if count >= min_freq]
    vocab = {word: i for i, word in enumerate(frequent_words)}

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary of {len(vocab)} words cached to {cache_path}")

    return vocab

def preprocess_corpora(config_path: str):
    """
    Builds a single vocabulary from a foundational corpus, then processes each
    corpus to create and cache training pairs.
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

    window_size = config.get("DEFAULT_WINDOW_SIZE", 2)
    cache_dir = config.get("CACHE_DIR", "cache")
    encoding = config.get("CORPUS_ENCODING", "utf-8")

    # 1. Initialize spaCy once
    initialize_nlp(config)

    # 2. Build one shared vocabulary from the foundational corpus
    foundational_corpus_path = config["FOUNDATIONAL_CORPUS_PATH"]
    vocab = build_vocab(foundational_corpus_path, config)
    print(f"--- Vocabulary loaded with {len(vocab)} words ---")

    # 3. Process each corpus to generate training pairs using the shared vocabulary
    print("\n--- Starting Training Pair Generation ---")
    for name, corpus_path in corpora_paths.items():
        print(f"\n>>> Processing corpus: '{name}'")

        data_path = os.path.join(cache_dir, f"{os.path.basename(corpus_path)}_ws{window_size}.pt")

        if os.path.exists(data_path):
            print(f"Pre-processed data already exists at {data_path}. Skipping.")
            continue
        
        print(f"Reading and tokenizing {corpus_path}...")
        with open(corpus_path, 'r', encoding=encoding) as f:
            text = f.read()
        
        nlp.max_length = len(text) + config.get("NLP_PADDING_SIZE", 100)
        doc = nlp(text)
        
        token_ids = [
            vocab[token.lemma_.lower()]
            for token in doc
            if not token.is_punct and not token.is_space and token.lemma_.lower() in vocab
        ]
        print(f"Corpus processed into {len(token_ids)} tokens.")

        print("Generating training pairs...")
        data_pairs = []
        for i, center_word_id in enumerate(token_ids):
            for w in range(1, window_size + 1):
                if i - w >= 0:
                    context_word_id = token_ids[i - w]
                    data_pairs.append((center_word_id, context_word_id))
                if i + w < len(token_ids):
                    context_word_id = token_ids[i + w]
                    data_pairs.append((center_word_id, context_word_id))

        if not data_pairs:
            print("Warning: No training pairs were generated for this corpus.")
            continue

        data_tensor = torch.tensor(data_pairs, dtype=torch.long)
        torch.save(data_tensor, data_path)
        print(f"Saved {len(data_tensor)} pairs to {data_path}")

    print("\n--- Pre-processing complete! ---")

if __name__ == '__main__':
    config_path = os.path.join('configs', 'base_config.yaml')
    preprocess_corpora(config_path)