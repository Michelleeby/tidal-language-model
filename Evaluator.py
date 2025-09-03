# Evaluator.py

import torch
import os
import json
import pickle

import torch.nn as nn

from typing import List, Dict, Tuple
from scipy.stats import spearmanr
from tqdm import tqdm

from TidalLanguageModel import TidalLanguageModel

class Evaluator:
    """
    Orchestrates the evaluation of a trained TidalLanguageModel, including:
    1.  Quantitative Benchmarks: Word similarity and analogy tasks.
    2.  Qualitative Analysis: Nearest neighbor generation under different tides.
    3.  N-Body Simulation: Visualization of semantic space dynamics.
    """

    def __init__(self, config: dict, experiment_dir: str, model_path: str, probe_words: List[str]):
        """
        Initializes the Evaluator.

        Args:
            config (dict): The experiment configuration.
            experiment_dir (str): Path to the experiment directory for saving results.
            model_path (str): Path to the trained model's state dictionary (.pth file).
            probe_words (List[str]): A list of words to use for qualitative analysis.
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.probe_words = probe_words
        self.device = torch.device(config.get("DEVICE", "cpu"))

        # Create results directory
        self.results_dir = os.path.join(experiment_dir, 'evaluation_results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Load vocabulary
        vocab_path = os.path.join(os.path.dirname(model_path), self.config.get("VOCAB_CACHE_PATH", "vocab.pkl"))
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.rev_vocab = {i: w for w, i in self.vocab.items()}
        
        # Load model using parameters from the config
        self.model = TidalLanguageModel(
            vocab_size=len(self.vocab),
            config=self.config,
            experiment_dir=self.experiment_dir
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Pre-calculate all semantic embeddings for efficiency
        with torch.no_grad():
            all_token_ids = torch.arange(len(self.vocab)).to(self.device)
            # The forward pass returns logits and loss, we need to get the embeddings
            # by calling the projection_layer directly on position_embeddings
            self.all_semantic_embeddings = self.model.projection_layer(self.model.position_embeddings(all_token_ids))


    def run(self):
        """
        Executes the full evaluation suite.
        """
        print("--- Running Full Evaluation Suite ---")
        self.run_quantitative_benchmarks()
        self.run_qualitative_analysis()
        self.run_and_log_simulations()
        print("\n--- Evaluation Complete ---")


    # ---------- Quantitative Benchmarks ----------

    def run_quantitative_benchmarks(self):
        """
        Runs all quantitative benchmarks and saves the results.
        """
        print("\n--- Running Quantitative Benchmarks ---")
        correlation = self._calculate_word_similarity_correlation()
        accuracy = self._solve_analogy_benchmark()

        results = {
            "word_similarity_spearman_correlation": correlation,
            "analogy_solving_accuracy": accuracy
        }

        print(f"Word Similarity Correlation: {correlation:.4f}")
        print(f"Analogy Solving Accuracy: {accuracy:.4f}")

        save_path = os.path.join(self.results_dir, 'quantitative_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Quantitative results saved to {save_path}")

    def _calculate_word_similarity_correlation(self) -> float:
        """
        Calculates the Spearman correlation between the model's embedding similarities
        and human-rated scores from the configuration benchmark.

        Returns:
            float: The Spearman correlation coefficient.
        """
        benchmark_data = self.config.get("WORD_SIMILARITY_BENCHMARK", [])
        if not benchmark_data:
            return 0.0

        model_scores = []
        human_scores = []

        for word1, word2, human_score in benchmark_data:
            if word1 in self.vocab and word2 in self.vocab:
                idx1 = self.vocab[word1]
                idx2 = self.vocab[word2]
                
                vec1 = self.all_semantic_embeddings[idx1].unsqueeze(0)
                vec2 = self.all_semantic_embeddings[idx2].unsqueeze(0)
                
                similarity = nn.functional.cosine_similarity(vec1, vec2).item()
                
                model_scores.append(similarity)
                human_scores.append(human_score)

        if len(model_scores) < 2:
            return 0.0
            
        correlation, _ = spearmanr(human_scores, model_scores)
        return correlation

    def _solve_analogy_benchmark(self) -> float:
        """
        Evaluates the model's ability to solve analogies (e.g., A:B::C:D).
        Calculates the accuracy of predicting 'D'.

        Returns:
            float: The accuracy (0.0 to 1.0) on the analogy benchmark.
        """
        benchmark_data = self.config.get("ANALOGY_BENCHMARK", [])
        if not benchmark_data:
            return 0.0

        correct_predictions = 0
        top_k = self.config.get("DEFAULT_TOP_K", 5)

        for analogy in benchmark_data:
            a, b, c, d_true = analogy
            
            if not all(word in self.vocab for word in analogy):
                continue
            
            vec_a = self.all_semantic_embeddings[self.vocab[a]]
            vec_b = self.all_semantic_embeddings[self.vocab[b]]
            vec_c = self.all_semantic_embeddings[self.vocab[c]]
            
            target_vec = vec_b - vec_a + vec_c
            
            neighbors = self._find_nearest_neighbors(target_vec, top_k=top_k)
            
            predicted_words = [n[0] for n in neighbors if n[0] not in [a, b, c]]

            if predicted_words and predicted_words[0] == d_true:
                correct_predictions += 1
                
        return correct_predictions / len(benchmark_data) if benchmark_data else 0.0

    # ---------- Qualitative Analysis ----------

    def run_qualitative_analysis(self):
        """
        Runs all qualitative analyses and saves the results.
        """
        print("\n--- Running Qualitative Analysis ---")
        all_associations = {}
        for word in self.probe_words:
            print(f"Generating associations for '{word}'...")
            associations = self._generate_tidal_word_associations(word)
            all_associations[word] = associations

        save_path = os.path.join(self.results_dir, 'qualitative_associations.json')
        with open(save_path, 'w') as f:
            json.dump(all_associations, f, indent=4)
        print(f"Qualitative results saved to {save_path}")


    def _generate_tidal_word_associations(self, probe_word: str, top_k: int = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        For a given probe word, finds the top_k nearest neighbors under different
        tidal conditions (Normal, High, Low, Storm).
        """
        if top_k is None:
            top_k = self.config.get("DEFAULT_TOP_K", 10)

        if probe_word not in self.vocab:
            return {}
            
        associations = {}
        tides = {"Normal": 0.0, "High": 1.0, "Low": -1.0, "Storm": None} # Storm is a special case

        with torch.no_grad():
            for tide_name, level in tides.items():
                self.model.set_tidal_level(level)
                
                all_token_ids = torch.arange(len(self.vocab)).to(self.device)
                current_embeddings = self.model.projection_layer(self.model.position_embeddings(all_token_ids))
                probe_vec = current_embeddings[self.vocab[probe_word]]
                
                original_embeddings = self.all_semantic_embeddings
                self.all_semantic_embeddings = current_embeddings
                
                neighbors = self._find_nearest_neighbors(probe_vec, top_k)
                associations[tide_name] = neighbors
                
                self.all_semantic_embeddings = original_embeddings

        self.model.set_tidal_level(None) # Reset to internal clock
        return associations


    def _find_nearest_neighbors(self, word_vec: torch.Tensor, top_k: int) -> List[Tuple[str, float]]:
        """
        Helper function to find the top_k most similar words from the vocabulary
        to a given word vector using cosine similarity.
        """
        if word_vec.dim() == 1:
            word_vec = word_vec.unsqueeze(0)

        similarities = nn.functional.cosine_similarity(word_vec, self.all_semantic_embeddings)
        top_k_plus_one = min(top_k + 1, len(self.vocab))
        
        top_scores, top_indices = torch.topk(similarities, top_k_plus_one)

        neighbors = []
        for score, idx in zip(top_scores, top_indices):
            word = self.rev_vocab[idx.item()]
            is_self = torch.all(self.all_semantic_embeddings[idx].eq(word_vec.squeeze(0)))
            if not is_self:
                neighbors.append((word, score.item()))

        return neighbors[:top_k]

    # ---------- N-Body Simulation & Data Logging ----------

    def run_and_log_simulations(self):
        """
        Runs long-form N-body simulations for each probe word and logs the
        time-series data to a file for later visualization.
        """
        print("\n--- Running N-Body Simulations for Data Logging ---")
        
        num_neighbors = self.config.get("SIMULATION_NUM_NEIGHBORS", 50)
        total_steps = self.config.get("SIMULATION_LOGGING_STEPS", self.config.get("CIRCADIAN_PERIOD", 2000) // 4)

        for probe_word in self.probe_words:
            if probe_word not in self.vocab:
                print(f"Probe word '{probe_word}' not in vocabulary. Skipping simulation.")
                continue
            
            print(f"Starting simulation for '{probe_word}'... ({total_steps} steps)")

            word_labels, token_ids = self._get_simulation_word_indices(probe_word, num_neighbors)
            token_ids = token_ids.to(self.device)

            simulation_log = []
            self.model.global_step.zero_() 

            for step in tqdm(range(total_steps), desc=f"Simulating '{probe_word}'"):
                positions_8d, hormones = self._run_simulation_step(token_ids)
                
                log_entry = {
                    'step': self.model.global_step.item(),
                    'tidal_level': self.model.get_tidal_level().item(),
                    'hormones': hormones,
                    'positions': positions_8d.cpu().tolist()
                }
                simulation_log.append(log_entry)

            output_data = {
                'probe_word': probe_word,
                'word_labels': word_labels,
                'simulation_log': simulation_log
            }
            
            sim_dir = os.path.join(self.results_dir, "simulation_logs")
            os.makedirs(sim_dir, exist_ok=True)
            save_path = os.path.join(sim_dir, f"simulation_log_{probe_word}.json")

            with open(save_path, 'w') as f:
                json.dump(output_data, f)
                
            print(f"Simulation data for '{probe_word}' saved to {save_path}")

    def _get_simulation_word_indices(self, probe_word: str, num_neighbors: int) -> Tuple[List[str], torch.Tensor]:
        """
        Selects the probe word and its closest neighbors for the simulation.
        """
        self.model.set_tidal_level(0.0) # Neutral tide for neighbor selection
        with torch.no_grad():
             all_embeds_512 = self.model.position_embeddings(torch.arange(len(self.vocab)).to(self.device))
             all_embeddings = self.model.projection_layer(all_embeds_512)
        
        probe_vec = all_embeddings[self.vocab[probe_word]]
        
        original_embeddings = self.all_semantic_embeddings
        self.all_semantic_embeddings = all_embeddings
        neighbors = self._find_nearest_neighbors(probe_vec, top_k=num_neighbors)
        self.all_semantic_embeddings = original_embeddings
        
        word_labels = [probe_word] + [n[0] for n in neighbors]
        token_ids = torch.tensor([self.vocab[w] for w in word_labels], dtype=torch.long)
        
        self.model.set_tidal_level(None) # Reset to internal clock
        return word_labels, token_ids

    def _run_simulation_step(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executes a single step of the physics simulation.
        """
        with torch.no_grad():
            self.model(token_ids) # This updates the internal state of the model
            
            # Retrieve updated positions and project to 8D
            updated_positions_512d = self.model.position_embeddings(token_ids)
            updated_positions_8d = self.model.projection_layer(updated_positions_512d)
        
        hormone_levels = self.model.endocrine_system.get_hormone_state()
        
        return updated_positions_8d, hormone_levels

