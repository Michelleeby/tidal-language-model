import torch
import os
import argparse
import pickle
import spacy
from ruamel.yaml import YAML

# Import the model class and other necessary components from your project structure
from TidalLanguageModel import TidalLanguageModel
from SequentialDataset import load_vocab
from Preprocess import initialize_nlp

# Global spaCy model
nlp = None

def tokenizer(text: str, vocab: dict, config: dict) -> torch.Tensor:
    """
    Converts a string of text into a tensor of token IDs using spaCy for lemmatization.
    """
    global nlp
    if nlp is None:
        # This function is defined in Preprocess.py
        initialize_nlp(config) 
        nlp = spacy.load(config.get("NLP_MODEL", "en_core_web_sm"), disable=config.get("NLP_DISABLE", ["parser", "ner"]))

    doc = nlp(text)
    token_ids = []
    for token in doc:
        lemma = token.lemma_.lower()
        if not token.is_punct and not token.is_space and lemma in vocab:
            token_ids.append(vocab[lemma])
        elif not token.is_punct and not token.is_space:
            print(f"  [Tokenizer Warning] OOV token: '{token.text}' (lemma: '{lemma}') - skipping.")
            
    return torch.tensor(token_ids, dtype=torch.long)

def detokenizer(token_ids: list, idx_to_vocab: dict) -> str:
    """
    Converts a list of token IDs back into a string.
    """
    tokens = [idx_to_vocab.get(idx, '[UNK]') for idx in token_ids]
    return ' '.join(tokens)

def run_generation(args):
    """
    Main function to load the model and run text generation.
    """
    print("--- Tidal Language Model Inference ---")
    
    # 1. Load Configuration
    print(f"Loading configuration from: {args.config}")
    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # 2. Load Vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab(config)
    idx_to_vocab = {i: word for word, i in vocab.items()}
    vocab_size = len(vocab)
    print(f"Vocabulary loaded with {vocab_size} words.")

    # 3. Initialize Model
    print("Initializing Tidal Language Model...")
    model = TidalLanguageModel(vocab_size=vocab_size, config=config, experiment_dir=os.path.dirname(args.checkpoint))
    
    # 4. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    print(f"Loading model state from: {args.checkpoint}")
    device = "cuda" if torch.cuda.is_available() and config['DEVICE'] != 'cpu' else "cpu"
    print(f"Loading model state from: {args.checkpoint}")
    device = "cuda" if torch.cuda.is_available() and config['DEVICE'] != 'cpu' else "cpu"
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded successfully.")

    # 5. Prepare Prompt
    print(f"\nOriginal Prompt: '{args.prompt}'")
    prompt_ids = tokenizer(args.prompt, vocab, config)
    if prompt_ids.nelement() == 0:
        print("Error: The prompt contains no words found in the vocabulary. Please try a different prompt.")
        return
    
    # 6. Run Generation with Different Tidal Settings
    tidal_settings = {
        "High Tide (Creative)": 1.0,
        "Neutral Tide": 0.0,
        "Low Tide (Constrained)": -1.0
    }

    for name, level in tidal_settings.items():
        print(f"\n--- Generating with {name} (tidal_level={level}) ---")
        
        generated_ids = model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            tidal_level=level,
            repetition_penalty=args.repetition_penalty
        )
        
        generated_text = detokenizer(generated_ids, idx_to_vocab)
        print(f"Generated Text:\n{generated_text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with the Tidal Language Model.")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth) file.')
    
    parser.add_argument('--prompt', type=str, required=True,
                        help='The starting text prompt for generation.')
                        
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum number of new tokens to generate.')
                        
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature. Higher is more random.')
                        
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k filtering for sampling.')
                        
    parser.add_argument('--config', type=str, default='base_config.yaml',
                        help='Path to the base configuration YAML file.')

    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                    help='Penalty for repeating tokens.')

    args = parser.parse_args()
    run_generation(args)
