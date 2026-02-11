"""
Evaluator.py

Evaluation suite for TransformerLM.
Computes perplexity on TinyStories validation set and generates sample text.
"""

import torch
import os
import json
import math
from typing import List

from torch.utils.data import DataLoader
from tqdm import tqdm

from TransformerLM import TransformerLM
from DataPipeline import TinyStoriesDataset, get_tokenizer


class Evaluator:
    """
    Evaluates a trained TransformerLM with:
    1. Perplexity on TinyStories validation set.
    2. Sample generation from a few prompts.
    """

    def __init__(
        self,
        config: dict,
        experiment_dir: str,
        model_path: str,
    ):
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = torch.device(config.get("DEVICE", "cpu"))
        if str(self.device) == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.results_dir = os.path.join(experiment_dir, "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.tokenizer = get_tokenizer()
        vocab_size = config.get("VOCAB_SIZE", self.tokenizer.vocab_size)

        self.model = TransformerLM(vocab_size=vocab_size, config=config)
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def run(self):
        """Execute the full evaluation suite."""
        print("--- Running Evaluation Suite ---")
        perplexity = self.compute_perplexity()
        samples = self.generate_samples()

        results = {
            "perplexity": perplexity,
            "samples": samples,
        }

        save_path = os.path.join(self.results_dir, "evaluation_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

        return results

    def compute_perplexity(self, max_batches: int = None) -> float:
        """
        Compute perplexity = exp(avg cross-entropy) on TinyStories validation set.

        Args:
            max_batches: Optional limit on number of batches (for quick eval).

        Returns:
            Perplexity value (float).
        """
        print("\n--- Computing Perplexity ---")
        max_length = self.config.get("MAX_CONTEXT_LENGTH", 256)
        batch_size = self.config.get("EVAL_BATCH_SIZE", self.config.get("BATCH_SIZE", 32))

        val_ds = TinyStoriesDataset(
            split="validation",
            max_length=max_length,
            tokenizer=self.tokenizer,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(self.config.get("NUM_CPU_CORE_WORKERS", 4), 4),
        )

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits, (loss, _), _ = self.model(input_ids, target_ids)

                batch_tokens = target_ids.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                num_batches += 1

                if max_batches and num_batches >= max_batches:
                    break

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss)

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Evaluated on {total_tokens:,} tokens ({num_batches} batches)")

        return perplexity

    def generate_samples(
        self,
        prompts: List[str] = None,
        max_new_tokens: int = 100,
    ) -> List[dict]:
        """
        Generate text from a set of prompts and display results.

        Args:
            prompts: List of prompt strings. Uses defaults if None.
            max_new_tokens: Number of tokens to generate per prompt.

        Returns:
            List of dicts with prompt and generated text.
        """
        print("\n--- Generating Samples ---")
        if prompts is None:
            prompts = [
                "Once upon a time",
                "The little cat",
                "Mom said to",
                "One day, a boy",
            ]

        samples = []
        for prompt in prompts:
            encoded = self.tokenizer.encode(prompt)
            prompt_ids = torch.tensor(encoded, dtype=torch.long)

            generated_ids = self.model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=40,
            )

            generated_text = self.tokenizer.decode(generated_ids)

            sample = {"prompt": prompt, "generated": generated_text}
            samples.append(sample)

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text[:200]}...")

        return samples
