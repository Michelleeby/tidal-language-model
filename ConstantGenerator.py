import torch
import os
import argparse
import pickle
import spacy
from ruamel.yaml import YAML

# Import the model class and other necessary components from your project structure
from ConstantLanguageModel import ConstantLanguageModel
from SequentialDataset import load_vocab
from Preprocess import initialize_nlp

# RL components (optional, for hormone-controlled generation)
try:
    from HormoneRLAgent import create_agent
    from HormoneModulator import HormoneModulator, RandomHormonePolicy, FixedHormonePolicy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Global spaCy model
nlp = None

def tokenizer(text: str, vocab: dict, config: dict) -> torch.Tensor:
    """
    Converts a string of text into a tensor of token IDs using spaCy tokenization.
    Note: No lemmatization is applied to match vocabulary building process.
    """
    global nlp
    if nlp is None:
        # This function is defined in Preprocess.py
        initialize_nlp(config)
        nlp = spacy.load(config.get("NLP_MODEL", "en_core_web_sm"), disable=config.get("NLP_DISABLE", ["parser", "ner"]))

    doc = nlp(text)
    token_ids = []
    for token in doc:
        word = token.text.lower()
        if not token.is_punct and not token.is_space and word in vocab:
            token_ids.append(vocab[word])
        elif not token.is_punct and not token.is_space:
            print(f"  [Tokenizer Warning] OOV token: '{token.text}' - skipping.")

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
    print("--- Constant Language Model Inference ---")

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
    print("Initializing Constant Language Model...")
    model = ConstantLanguageModel(vocab_size=vocab_size, config=config, experiment_dir=os.path.dirname(args.checkpoint))

    # 4. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
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

    # 6. Run Generation
    print(f"\n--- Generating Text ---")

    # Check if RL-controlled generation is requested
    if args.rl_agent and RL_AVAILABLE:
        print("Using RL-controlled hormone generation...")

        # Load RL config
        rl_config_path = args.rl_config or 'configs/rl_config.yaml'
        if os.path.exists(rl_config_path):
            with open(rl_config_path, 'r') as f:
                rl_config = yaml.load(f)
            # Merge with base config
            config.update(rl_config)

        # Create modulator
        modulator = HormoneModulator(config)

        # Load or create RL agent
        if args.rl_checkpoint:
            print(f"Loading RL agent from: {args.rl_checkpoint}")
            agent = create_agent(config, device)
            checkpoint = torch.load(args.rl_checkpoint, map_location=device)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            agent.eval()
            hormone_policy = agent
        elif args.hormone_mode == "random":
            print("Using random hormone policy")
            hormone_policy = RandomHormonePolicy(device)
        elif args.hormone_mode == "fixed":
            print(f"Using fixed hormone policy: catalyst={args.catalyst}, stress={args.stress}, inhibitor={args.inhibitor}")
            hormone_policy = FixedHormonePolicy(
                catalyst=args.catalyst,
                stress=args.stress,
                inhibitor=args.inhibitor,
                device=device
            )
        else:
            print("Using learned RL agent (default checkpoint)")
            # Try to find checkpoint in experiment directory
            checkpoint_path = os.path.join(os.path.dirname(args.checkpoint), "rl_checkpoint_final.pth")
            if os.path.exists(checkpoint_path):
                agent = create_agent(config, device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                agent.load_state_dict(checkpoint["agent_state_dict"])
                agent.eval()
                hormone_policy = agent
            else:
                print(f"Warning: No RL checkpoint found at {checkpoint_path}, using fixed hormones")
                hormone_policy = FixedHormonePolicy(device=device)

        # Generate with hormones
        generated_ids, trajectory = model.generate_with_hormones(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            hormone_policy=hormone_policy,
            modulator=modulator,
            base_temperature=args.temperature,
            top_k=args.top_k,
            return_trajectory=args.verbose
        )

        # Print hormone trajectory if verbose
        if args.verbose and trajectory:
            print("\n--- Hormone Trajectory ---")
            for i, action in enumerate(trajectory["actions"]):
                if i < 10 or i >= len(trajectory["actions"]) - 5:  # First 10 and last 5
                    print(f"  Step {i}: catalyst={action[0]:.3f}, stress={action[1]:.3f}, inhibitor={action[2]:.3f}")
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
            tidal_level=None,  # Not used by constant model
            repetition_penalty=args.repetition_penalty
        )

    generated_text = detokenizer(generated_ids, idx_to_vocab)
    print(f"Generated Text:\n{generated_text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with the Constant Language Model.")

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

    parser.add_argument('--config', type=str, default='configs/constant_base_config.yaml',
                        help='Path to the base configuration YAML file.')

    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                    help='Penalty for repeating tokens.')

    # RL Hormone Control arguments
    parser.add_argument('--rl-agent', action='store_true',
                        help='Use RL-controlled hormone generation.')

    parser.add_argument('--rl-checkpoint', type=str, default=None,
                        help='Path to trained RL agent checkpoint.')

    parser.add_argument('--rl-config', type=str, default=None,
                        help='Path to RL configuration file.')

    parser.add_argument('--hormone-mode', type=str, default='learned',
                        choices=['learned', 'random', 'fixed'],
                        help='Hormone control mode: learned (RL agent), random, or fixed.')

    parser.add_argument('--catalyst', type=float, default=0.5,
                        help='Fixed catalyst hormone level (0-1). Used with --hormone-mode fixed.')

    parser.add_argument('--stress', type=float, default=0.5,
                        help='Fixed stress hormone level (0-1). Used with --hormone-mode fixed.')

    parser.add_argument('--inhibitor', type=float, default=0.5,
                        help='Fixed inhibitor hormone level (0-1). Used with --hormone-mode fixed.')

    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed hormone trajectory during generation.')

    args = parser.parse_args()
    run_generation(args)
