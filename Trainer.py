import torch
import torch.optim as optim
import torch.nn as nn
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from SequentialDataset import load_or_create_dataset, load_vocab
from Utils import setup_logger, plot_semantic_space, format_cluster_analysis_text
from TidalLanguageModel import TidalLanguageModel
from DynamicLRScheduler import DynamicLRScheduler

class Trainer:
    """
    Handles the entire training pipeline for the Tidal Language Model,
    including early stopping, checkpointing, and resuming from checkpoints.
    """
    def __init__(self, config, experiment_dir, shutdown_event: threading.Event):
        self.config = config
        self.exp_dir = experiment_dir
        self.shutdown_event = shutdown_event
        
        self.device = self._get_device()
        self.logger = setup_logger('Training', os.path.join(self.exp_dir, 'training.log'), config)
        self.logger_visualization = setup_logger('Visualization', os.path.join(self.exp_dir, 'visualization.log'), config)
        
        self.model = None
        self.optimizer = None
        self.criterion = None

        # Add the TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard_logs'))

        self.viz_enabled = self.config.get("ENABLE_VISUALIZATION", True)
        if self.viz_enabled:
            self.frames_dir = os.path.join(self.exp_dir, self.config.get("TRAINING_FRAMES_SUBDIR", "semantic_space_frames"))
            os.makedirs(self.frames_dir, exist_ok=True)

        self.current_epoch_num = 0
        # Initialize a thread pool with one worker for async I/O operations.
        # This prevents plotting and file saving from blocking the main training thread.
        # Making it a single worker is important for sequential queuing.
        self.visualization_executor = ThreadPoolExecutor(max_workers=1)
        self.last_viz_time = 0
        self.tags = self.config.get("TENSORBOARD_TAGS", {})

    def _log_and_save_visuals_async(self, viz_data, global_step, batch_tokens, seq_len, vocab, tide_name):
        """
        Handles all slow I/O (plotting, file saving, text formatting) in a separate thread.
        """
        self.logger_visualization.info(f"Logging and saving visuals for global step {global_step}...")

        fig, ax = plt.subplots(figsize=self.config.get("TRAINING_PLOT_FIGSIZE", (8, 8)))
        viz_data_sliced = {
            'positions_2d': viz_data['positions_2d'][:seq_len],
            'positions_8d': viz_data['positions_8d'][:seq_len],
            'forces_2d': viz_data['forces_2d'][:seq_len],
            'masses': viz_data['masses'][:seq_len]
        }
        token_ids_for_viz = batch_tokens[0]

        kmeans = None
        try:
            kmeans = plot_semantic_space(
                fig, 
                ax,
                viz_data_sliced,
                token_ids_for_viz,
                vocab,
                self.probe_words,
                self.current_epoch_num,
                global_step % len(self.current_dataloader),
                title_suffix=f"({tide_name.capitalize()} Tide)" if tide_name else "",
                config=self.config
            )
        except Exception as e:
            self.logger_visualization.error(f"Error plotting semantic space: {e}")
            plt.close(fig)
            return

        if kmeans is None:
            self.logger_visualization.error("KMeans clustering failed. Skipping visualization.")
            return

        frame_path = os.path.join(self.frames_dir, f'frame_{global_step:06d}.png')
        fig.savefig(frame_path, dpi=100)
        self.writer.add_figure('Semantic_Space', fig, global_step)
        plt.close(fig)
        self.logger_visualization.info("Successfully saved visualization.")
        
        try:
            cluster_text = format_cluster_analysis_text(kmeans.cluster_centers_, self.config)
            self.writer.add_text('Cluster_Analysis', cluster_text, global_step)
        except Exception as e:
            self.logger_visualization.error(f"Error saving cluster analysis: {e}")
            return

        self.logger_visualization.info("Successfully saved cluster analysis.")
    
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
    
    def _train_epoch(self, data_loader, tide_name=None, vocab=None):
        """Generic training function for one epoch."""
        self.model.train()
        self.current_dataloader = data_loader
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch_num} Training")
        viz_interval_seconds = self.config.get("VISUALIZATION_INTERVAL_SECONDS", 60)
        for i, (input_sequence, target_sequence) in enumerate(progress_bar):
            if self.shutdown_event.is_set():
                self.logger.info("Shutdown detected, stopping training loop.")
                break
            
            input_sequence_gpu = input_sequence.to(self.device)
            target_sequence_gpu = target_sequence.to(self.device)
            self.optimizer.zero_grad()
            logits, physics_loss, viz_data = self.model(input_sequence_gpu, target_sequence_gpu)
            # Logits shape: [B, S, Vocab_Size] -> [B * S, Vocab_Size]
            # Target shape: [B, S] -> [B * S]
            vocab_size = logits.shape[-1]
            prediction_loss = self.criterion(logits.view(-1, vocab_size), target_sequence_gpu.view(-1))
            # Prepare scalar versions of losses for logging before combining for backprop.
            mean_physics_loss = physics_loss.mean()
            scalar_physics_loss = mean_physics_loss.item()
            prediction_loss_item = prediction_loss.item()
            loss = prediction_loss + self.config["PHYSICS_LOSS_WEIGHT"] * physics_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["MAX_GRAD_NORM"])
            self.optimizer.step()
            
            # Update learning rate on each step
            if self.scheduler:
                self.scheduler.step(tide_name=tide_name)

            total_loss += loss.item()
            tags = self.tags
            current_time = time.time()
            if self.viz_enabled and (current_time - self.last_viz_time > viz_interval_seconds):
                self.last_viz_time = current_time # Update the timer immediately

                global_step = self.current_epoch_num * len(data_loader) + i

                # Fast, synchronous logging remains on the main thread, 
                # as the negligible performance impact is a worthwhile trade-off 
                # for ensuring the sequential integrity of the log data without 
                # introducing asynchronous complexity.
                loss_metrics = {
                    'Total': loss.item(), 
                    'Prediction': prediction_loss_item, 
                    'Physics': scalar_physics_loss,
                }
                self.writer.add_scalars(tags["LOSSES"], loss_metrics, global_step)

                if self.model.enable_endocrine_system:
                    self.writer.add_scalars(tags["HORMONE_LEVELS"], self.model.endocrine_system.get_hormone_state(), global_step)
                
                physics_params = {
                    'G': self.model.physics_simulator.G.item(), 
                    'Repulsion_Strength': self.model.physics_simulator.repulsion_strength.item(), 
                    'Well_Attraction': self.model.physics_simulator.well_attraction_strength.item(), 
                    'Temperature': self.model.physics_simulator.temperature.item()
                }
                self.writer.add_scalars(tags["PHYSICS_PARAMS"], physics_params, global_step)

                # Force the writer to save the buffered data to disk immediately.
                # This ensures the dashboard always has access to the freshest data.
                self.writer.flush()

                # Slow I/O tasks are offloaded to the background thread, so this call returns immediately.
                # NOTE: We must pass all data to this new thread as CPU tensors or NumPy arrays by using .detach().cpu().
                # This avoids issues with GPU contexts and to prevent holding onto the computation graph.
                self.visualization_executor.submit(
                    self._log_and_save_visuals_async,
                    viz_data,              # Already detached in the model forward pass.
                    global_step,
                    input_sequence.cpu(),
                    input_sequence.shape[1],
                    vocab,
                    tide_name
                )

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(tags["LEARNING_RATE"], current_lr, self.scheduler.current_step)

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        self.current_epoch_num += 1
        return avg_loss

    def _save_checkpoint(self, epoch, phase_name):
        """Saves the model's state as a checkpoint with a standardized name."""
        phase_name_slug = phase_name.lower().replace("-", "_")
        checkpoint_name = f"checkpoint_{phase_name_slug}_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.exp_dir, checkpoint_name)
        self.logger.info(f"Caching progress. Saving checkpoint to {checkpoint_path}")
        state_dict = self.model._orig_mod.state_dict() if hasattr(self.model, '_orig_mod') else self.model.state_dict()
        torch.save(state_dict, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """Loads a model state from a checkpoint file."""
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            model_to_load = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            model_to_load.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
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

    def _training_loop(self, data_loader, phase_name, max_epochs, start_epoch=0, cache_freq=None, cache_milestones=None, vocab=None):
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
            avg_loss = self._train_epoch(data_loader, vocab=vocab)
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
            name: self._get_data_loader(load_or_create_dataset(path, self.config))
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
                avg_loss = self._train_epoch(loader, tide_name=tide_name, vocab=vocab)
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

    def shutdown(self):
        """Shuts down the internal thread pool executor."""
        self.logger.info("Shutting down visualization executor...")
        self.visualization_executor.shutdown(wait=True, cancel_futures=False)
        self.writer.close()
        self.logger.info("Trainer cleanup complete.")

    def run(self):
        """Main function to orchestrate the model training pipeline."""
        self.logger.info("Building vocabulary...")
        vocab = load_vocab(self.config)
        self.probe_words = self.config.get("PROBE_WORDS", [])
        vocab_size = len(vocab)

        # This is an estimate, but it's crucial for the cosine annealing schedule.
        max_foundational_epochs = self.config["NUM_EPOCHS"]
        # Use a dummy dataset to find the number of batches.
        temp_dataset = load_or_create_dataset(self.config["FOUNDATIONAL_CORPUS_PATH"], self.config)
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
            dataset = load_or_create_dataset(self.config["FOUNDATIONAL_CORPUS_PATH"], self.config)
            loader = self._get_data_loader(dataset)
            self._training_loop(loader, foundational_phase_name, max_foundational_epochs, start_epoch_foundational, cache_milestones=[max_foundational_epochs // 2], vocab=vocab)

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
                    dataset = load_or_create_dataset(corpus_path, self.config)
                    loader = self._get_data_loader(dataset)
                    self._training_loop(loader, early_tune_phase_name, early_tune_epochs, start_epoch_early, vocab=vocab)
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

        if not self.shutdown_event.is_set():
            self.logger.info(f"\n--- Training complete. Saving final model to {final_model_path} ---")
            torch.save(self.model.state_dict(), final_model_path)

            self.logger.info("Saving final embeddings for TensorBoard Projector.")
            final_embeddings_512d = self.model.position_embeddings.weight.detach()
            final_embeddings_8d = self.model.projection_layer(final_embeddings_512d).cpu()
            idx_to_word = {v: k for k, v in vocab.items()}
            metadata = [idx_to_word.get(i, '<UNK>') for i in range(len(vocab))]
            self.writer.add_embedding(
                mat=final_embeddings_8d,
                metadata=metadata,
                tag='Semantic_Space_8D'
            )
            return final_model_path
        else:
            self.logger.info("\n--- Training interrupted. Final model not saved. ---")
            return None
