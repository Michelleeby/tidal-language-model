import torch
import torch.optim as optim
import torch.nn as nn
import os
import re
import glob
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from AssociativeDataset import load_or_create_dataset, load_vocab
from Utils import setup_logger, plot_semantic_space
from TidalLanguageModel import TidalLanguageModel
from DynamicLRScheduler import DynamicLRScheduler

class Trainer:
    """
    Handles the entire training pipeline for the Tidal Language Model,
    including early stopping, checkpointing, and resuming from checkpoints.
    """
    def __init__(self, config, experiment_dir):
        self.config = config
        self.exp_dir = experiment_dir
        self.device = self._get_device()
        self.logger = setup_logger('Training', os.path.join(self.exp_dir, 'training.log'), config)
        
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.viz_enabled = self.config.get("ENABLE_VISUALIZATION", True)
        if self.viz_enabled:
            plt.ion() # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.current_epoch_num = 0

    def _get_device(self):
        """Determines the torch device based on config and availability."""
        if self.config['DEVICE'] == 'auto':
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config['DEVICE']

    def _setup_model(self, vocab_size, total_foundational_steps=1):
        """Instantiates the model, loss function, optimizer, and scheduler."""
        self.logger.info("Setting up model, optimizer, criterion, and scheduler...")
        self.model = TidalLanguageModel(vocab_size=vocab_size, config=self.config, experiment_dir=self.exp_dir)
        self.model.to(self.device)
        self.model = torch.compile(self.model) # Just-in-time (JIT) compiler. Analyze code and fuse operations into more efficient kernels.
        self.criterion = nn.CrossEntropyLoss()
        
        # Use the base_lr from the scheduler config for the optimizer
        base_lr = self.config.get("LEARNING_RATE_SCHEDULER", {}).get("BASE_LR", 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
        
        # Instantiate the scheduler
        self.scheduler = DynamicLRScheduler(self.optimizer, self.config, total_foundational_steps)
        
    def _get_data_loader(self, dataset):
        """Creates a DataLoader for a given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.config.get("NUM_CPU_CORE_WORKERS", 4),
            pin_memory=True,
            persistent_workers=True
        )
    
    def _train_epoch(self, data_loader, tide_name=None):
        """Generic training function for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch_num} Training")
        viz_freq = self.config.get("VISUALIZATION_FREQUENCY", 100)
        for i, (center_words, context_words) in enumerate(progress_bar):
            center_words, context_words = center_words.to(self.device), context_words.to(self.device)
            self.optimizer.zero_grad()
            logits, physics_loss, viz_data = self.model(center_words, context_words)
            prediction_loss = self.criterion(logits, context_words)
            loss = prediction_loss + self.config["PHYSICS_LOSS_WEIGHT"] * physics_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["MAX_GRAD_NORM"]) # Clip gradients to prevent exploding gradients.
            self.optimizer.step()
            
            # Update learning rate on each step
            if self.scheduler:
                self.scheduler.step(tide_name=tide_name)

            total_loss += loss.item()

            if self.viz_enabled and i % viz_freq == 0:
                plot_semantic_space(
                    self.ax,
                    viz_data['positions_2d'],
                    viz_data['forces_2d'],
                    self.current_epoch_num,
                    i,
                    title_suffix=f"({tide_name.capitalize()} Tide)" if tide_name else ""
                )
                plt.pause(0.001) # Give the plot a moment to update and process events
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        self.current_epoch_num += 1
        return avg_loss

    def _save_checkpoint(self, epoch, phase_name):
        """Saves the model's state as a checkpoint with a standardized name."""
        phase_name_slug = phase_name.lower().replace("-", "_")
        checkpoint_name = f"checkpoint_{phase_name_slug}_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.exp_dir, checkpoint_name)
        self.logger.info(f"Caching progress. Saving checkpoint to {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """Loads a model state from a checkpoint file."""
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.logger.info("Successfully loaded checkpoint.")
        except FileNotFoundError:
            self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def _find_latest_checkpoint(self, phase_name):
        """Finds the latest checkpoint file for a given phase."""
        phase_name_slug = phase_name.lower().replace("-", "_")
        search_pattern = os.path.join(self.exp_dir, f"checkpoint_{phase_name_slug}_epoch_*.pth")
        checkpoint_files = glob.glob(search_pattern)
        
        if not checkpoint_files:
            return None, 0

        latest_epoch = -1
        latest_checkpoint = None
        
        for f_path in checkpoint_files:
            basename = os.path.basename(f_path)
            match = re.search(r'epoch_(\d+).pth', basename)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = f_path
                    
        if latest_checkpoint:
            self.logger.info(f"Found latest checkpoint for phase '{phase_name}' at epoch {latest_epoch}.")
            return latest_checkpoint, latest_epoch
            
        return None, 0

    def _training_loop(self, data_loader, phase_name, max_epochs, start_epoch=0, cache_freq=None, cache_milestones=None):
        """
        Manages a standard training loop for a given phase with early stopping.
        """
        self.scheduler.set_phase("foundational")
        self.logger.info(f"--- Starting {phase_name} Phase (resuming from epoch {start_epoch}) ---")
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config.get("PATIENCE", 5)
        min_delta = self.config.get("MIN_DELTA", 0.0001)

        last_epoch_num = start_epoch
        for epoch in range(start_epoch, max_epochs):
            epoch_num = epoch + 1
            last_epoch_num = epoch_num
            avg_loss = self._train_epoch(data_loader)
            self.logger.info(f"{phase_name} Epoch {epoch_num}/{max_epochs} - Average Loss: {avg_loss:.4f}")

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(f"No significant improvement. Patience counter: {patience_counter}/{patience}")

            if (cache_freq and epoch_num % cache_freq == 0) or \
               (cache_milestones and epoch_num in cache_milestones):
                self._save_checkpoint(epoch_num, phase_name)

            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch_num}.")
                break
        
        self._save_checkpoint(last_epoch_num, phase_name)

    def _tidal_fine_tuning_loop(self, vocab, phase_name, max_epochs, start_epoch=0, cache_freq=5):
        """
        Manages the fine-tuning loop across multiple tidal corpora within each epoch.
        """
        self.scheduler.set_phase("finetuning")
        self.logger.info(f"--- Starting {phase_name} Phase (resuming from epoch {start_epoch}) ---")

        tide_paths = {
            "high": self.config.get("HIGH_TIDE_CORPUS_PATH"),
            "low": self.config.get("LOW_TIDE_CORPUS_PATH"),
            "storm": self.config.get("STORM_TIDE_CORPUS_PATH"),
        }

        tidal_loaders = {
            name: self._get_data_loader(load_or_create_dataset(path, vocab, self.config))
            for name, path in tide_paths.items() if path and os.path.exists(path)
        }

        if not tidal_loaders:
            self.logger.warning("No valid tidal corpus paths found in config. Skipping fine-tuning.")
            return
            
        tide_states = {
            tide_name: {'best_loss': float('inf'), 'patience_counter': 0, 'is_active': True}
            for tide_name in tidal_loaders.keys()
        }

        patience = self.config.get("PATIENCE", 5)
        min_delta = self.config.get("MIN_DELTA", 0.0001)

        last_epoch_num = start_epoch
        for epoch in range(start_epoch, max_epochs):
            epoch_num = epoch + 1
            last_epoch_num = epoch_num
            self.logger.info(f"--- {phase_name} Epoch {epoch_num}/{max_epochs} ---")

            if not any(state['is_active'] for state in tide_states.values()):
                self.logger.info("All tides have stabilized. Ending fine-tuning early.")
                break

            for tide_name, loader in tidal_loaders.items():
                state = tide_states[tide_name]
                if not state['is_active']:
                    continue

                tidal_level_map = {"high": 1.0, "low": -1.0, "storm": 0.0}
                self.model.set_tidal_level(tidal_level_map.get(tide_name))
                self.logger.info(f"Training on {tide_name} tide...")
                avg_loss = self._train_epoch(loader, tide_name=tide_name)
                self.logger.info(f"  > {tide_name.capitalize()} Tide Loss: {avg_loss:.4f}")

                if state['best_loss'] - avg_loss > min_delta:
                    state['best_loss'] = avg_loss
                    state['patience_counter'] = 0
                else:
                    state['patience_counter'] += 1
                    self.logger.info(f"  > No improvement for {tide_name} tide. Patience: {state['patience_counter']}/{patience}")

                if state['patience_counter'] >= patience:
                    self.logger.info(f"{tide_name.capitalize()} tide has stabilized and will be skipped.")
                    state['is_active'] = False

            self.model.set_tidal_level(None)

            if cache_freq and epoch_num % cache_freq == 0:
                self._save_checkpoint(epoch_num, phase_name)
        
        self._save_checkpoint(last_epoch_num, phase_name)

    def run(self):
        """Main function to orchestrate the model training pipeline."""
        self.logger.info("Building vocabulary...")
        vocab = load_vocab(self.config)
        vocab_size = len(vocab)

        # This is an estimate, but it's crucial for the cosine annealing schedule.
        max_foundational_epochs = self.config["NUM_EPOCHS"]
        # Use a dummy dataset to find the number of batches.
        temp_dataset = load_or_create_dataset(self.config["FOUNDATIONAL_CORPUS_PATH"], vocab, self.config)
        num_batches = len(temp_dataset) // self.config["BATCH_SIZE"]
        total_foundational_steps = max_foundational_epochs * num_batches

        self._setup_model(vocab_size, total_foundational_steps)
        
        # Foundational Pre-training
        self.logger.info("Disabling endocrine system and resonance for foundational training.")
        self.model.enable_endocrine_system = False
        self.model.enable_resonance = False
        
        foundational_phase_name = "Foundational"
        checkpoint_path, start_epoch_foundational = self._find_latest_checkpoint(foundational_phase_name)
        if checkpoint_path: self._load_checkpoint(checkpoint_path)

        max_foundational_epochs = self.config["NUM_EPOCHS"]
        if start_epoch_foundational >= max_foundational_epochs:
            self.logger.info("Foundational training already completed. Skipping.")
        else:
            dataset = load_or_create_dataset(self.config["FOUNDATIONAL_CORPUS_PATH"], vocab, self.config)
            loader = self._get_data_loader(dataset)
            self._training_loop(loader, foundational_phase_name, max_foundational_epochs, start_epoch_foundational, cache_milestones=[max_foundational_epochs // 2])

        # Fine-tuning
        self.logger.info("Re-enabling endocrine system and resonance for fine-tuning.")
        self.model.enable_endocrine_system = True
        self.model.enable_resonance = True

        # Early Fine-Tuning (Optional)
        early_tune_epochs = self.config.get("EARLY_TUNE_EPOCHS")
        if early_tune_epochs:
            early_tune_phase_name = "Early-Fine-tuning"
            checkpoint_path, start_epoch_early = self._find_latest_checkpoint(early_tune_phase_name)
            if checkpoint_path: self._load_checkpoint(checkpoint_path)

            if start_epoch_early >= early_tune_epochs:
                self.logger.info("Early fine-tuning already completed. Skipping.")
            else:
                corpus_path = self.config.get("EARLY_TUNE_CORPUS_PATH") or self.config.get("HIGH_TIDE_CORPUS_PATH")
                if corpus_path:
                    dataset = load_or_create_dataset(corpus_path, vocab, self.config)
                    loader = self._get_data_loader(dataset)
                    self._training_loop(loader, early_tune_phase_name, early_tune_epochs, start_epoch_early)
                else:
                    self.logger.warning("EARLY_TUNE_EPOCHS specified but no corpus path found. Skipping.")

        # Tidal Fine-Tuning
        tidal_tune_phase_name = "Tidal-Fine-tuning"
        checkpoint_path, start_epoch_tidal = self._find_latest_checkpoint(tidal_tune_phase_name)
        if checkpoint_path: self._load_checkpoint(checkpoint_path)

        max_finetune_epochs = self.config["FINE_TUNE_EPOCHS"]
        if start_epoch_tidal >= max_finetune_epochs:
            self.logger.info("Tidal fine-tuning already completed. Skipping.")
        else:
            self._tidal_fine_tuning_loop(vocab, tidal_tune_phase_name, max_finetune_epochs, start_epoch_tidal)

        final_model_path = os.path.join(self.exp_dir, f"{self.config['TIDAL_MODEL_NAME']}_v{self.config['TIDAL_MODEL_VERSION']}.pth")
        self.logger.info(f"\n--- Training complete. Saving final model to {final_model_path} ---")
        torch.save(self.model.state_dict(), final_model_path)
        
        if self.viz_enabled:
            self.logger.info("Training finished. Displaying final semantic space.")
            plt.ioff() # Turn off interactive mode
            self.fig.savefig(os.path.join(self.exp_dir, "final_semantic_space.png")) # Save the final plot
            plt.show() # Show the plot and wait for user to close
    
        return final_model_path
