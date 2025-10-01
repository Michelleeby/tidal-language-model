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
import wandb
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm

from SequentialDataset import load_or_create_dataset, load_vocab
from Utils import setup_logger, plot_semantic_space, format_cluster_analysis_text
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
        self.logger, self.logger_visualization = self._setup_loggers(config.copy())
        self.last_viz_time = 0
        self.qualitative_log_buffer = []
        self.pending_viz_futures = []

        # Initialize a thread pool with one worker for async I/O operations.
        # This prevents plotting and file saving from blocking the main training thread.
        # Making it a single worker is important for simple, sequential queuing.
        self.visualization_executor = ThreadPoolExecutor(max_workers=1)
                
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler()
        self.current_epoch_num = 0
        
        self.tags = self.config.get("TENSORBOARD_TAGS", {})

    def _log_qualitative_events_async(self, log_entries, idx_to_word, global_step):
        """Processes a batch of trigger events and logs them as a wandb.Table."""
        if not log_entries:
            return

        try:
            columns = [
                "Global Step", "Sequence Text", "Trigger Name", "Hormone", 
                "Strength", "Threshold", "Status", "Tidal Level"
            ]
            table_data = []

            for entry in log_entries:
                tokens = [idx_to_word.get(idx, "<UNK>") for idx in entry["token_ids"]]
                text = " ".join(tokens)
                strength = entry["strength"]
                threshold = entry["threshold"]
                status = "🔥 ACTIVE" if strength > threshold else "dormant"
                
                table_data.append([
                    global_step,
                    text,
                    entry["trigger_name"],
                    entry["hormone_name"],
                    f"{strength:.4f}",
                    f"{threshold:.2f}",
                    status,
                    f"{entry['tidal_level']:.2f}"
                ])
            
            log_dict = {
                "Trigger Analysis Events": wandb.Table(data=table_data, columns=columns)
            },

            return log_dict

        except Exception as e:
            self.logger_visualization.error(f"Error during async qualitative logging: {e}", exc_info=True)

    def _log_visuals_async(self, viz_data, global_step, batch_tokens, seq_len, vocab, tide_name):
        """Logs visualizations to W&B in a separate thread to avoid blocking training."""
        try:
            fig, ax = plt.subplots(figsize=self.config.get("TRAINING_PLOT_FIGSIZE", (8, 8)))    
            viz_data_sliced = {
                'positions_2d': viz_data['positions_2d'][:seq_len],
                'positions_8d': viz_data['positions_8d'][:seq_len],
                'forces_2d': viz_data['forces_2d'][:seq_len],
                'masses': viz_data['masses'][:seq_len]
            }
            token_ids_for_viz = batch_tokens[0]
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

            if kmeans is None:
                self.logger_visualization.error("KMeans clustering failed, skipping visualization log.")
                return

            cluster_text = format_cluster_analysis_text(kmeans.cluster_centers_, self.config)
            
            log_dict = {
                "Semantic Space": wandb.Image(fig),
                "Cluster Analysis": cluster_text
            }

            return log_dict

        except Exception as e:
            self.logger_visualization.error(f"Error during async visualization: {e}", exc_info=True)
        finally:
            if 'fig' in locals() and fig:
                plt.close(fig)

    def _log_metrics(self, data, global_step):
        # Fast, synchronous logging remains on the main thread, 
        # as the negligible performance impact is a worthwhile trade-off 
        # for ensuring the sequential integrity of the log data without 
        # introducing asynchronous complexity.
        metrics_dict = {
            "Losses/Total": data['total_loss'],
            "Losses/Prediction": data['prediction_loss'],
            "Losses/Physics": data['physics_loss'],
            "Energy/F_pos": data['physics_loss_components']['F_pos'],
            "Energy/F_neg": data['physics_loss_components']['F_neg'],
            "Physics/G": self.model.physics_simulator.G.item(),
            "Physics/Repulsion_Strength": self.model.physics_simulator.repulsion_strength.item(),
            "Physics/Well_Attraction": self.model.physics_simulator.well_attraction_strength.item(),
            "Physics/Temperature": self.model.physics_simulator.temperature.item(),
            "Learning Rate": self.optimizer.param_groups[0]['lr']
        }

        if self.model.enable_endocrine_system:
            hormone_state = self.model.endocrine_system.get_hormone_state()
            hormone_logs = {f"Hormones/{k}": v for k, v in hormone_state.items()}
            metrics_dict.update(hormone_logs)

        # Create a new list to hold futures that are not yet complete.
        still_pending_futures = []
        # Process all pending futures in a single loop.
        for future in self.pending_viz_futures:
            
            # If a future is not done, keep it for the next check.
            if not future.done():
                still_pending_futures.append(future)
                continue
            
            # If the future is done, try to get its result and log it.
            try:
                result = future.result()
                if result:
                    metrics_dict.update(result)
            except Exception as e:
                self.logger_visualization.error(f"Async logging task failed: {e}", exc_info=True)

        # Atomically replace the old list with the new one.
        self.pending_viz_futures = still_pending_futures

        wandb.log(metrics_dict, step=global_step)

        current_time = time.time()
        viz_interval_seconds = self.config.get("VISUALIZATION_INTERVAL_SECONDS", 60)
        if current_time - self.last_viz_time > viz_interval_seconds:
            self.last_viz_time = current_time
            
            # Slow I/O tasks are offloaded to the background visualization thread, so this call returns immediately.
            # NOTE: We must pass all data to this new thread as CPU tensors or NumPy arrays by using .detach().cpu().
            # This avoids issues with GPU contexts and to prevent holding onto the computation graph.
            future = self.visualization_executor.submit(
                self._log_visuals_async,
                data['viz_data'], 
                global_step, 
                data['input_sequence'], # Already on CPU from _train_epoch.
                data['input_sequence_shape'], 
                data['vocab'], 
                data['tide_name']
            )
            self.pending_viz_futures.append(future)
        
        qualitative_log_interval = self.config.get("QUALITATIVE_LOG_INTERVAL_BATCHES", 100)
        if global_step % qualitative_log_interval == 0 and self.qualitative_log_buffer:
            # 1. Atomically copy and clear the buffer.
            logs_to_process = self.qualitative_log_buffer.copy()
            self.qualitative_log_buffer.clear()
            
            # 2. Submit the processing task to the background thread.
            future = self.visualization_executor.submit(
                self._log_qualitative_events_async,
                logs_to_process,
                self.idx_to_word,
                data['vocab'],
                global_step
            )
            self.pending_viz_futures.append(future)

    def _get_device(self):
        """Determines the torch device based on config and availability."""
        if self.config['DEVICE'] == 'auto':
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config['DEVICE']

    def _setup_loggers(self, config):
        logger_types = config.get('TRAINER_LOGGER_TYPES', ['training', 'visualization'])
        logger_configs = {
            logger_type: {
                'log_file': os.path.join(self.exp_dir, f'{logger_type.lower()}.log'),
                'config': config.copy()
            }
            for logger_type in logger_types
        }
        logger_configs['training']['config']['ENABLE_CONSOLE_LOGGING'] = True
        loggers = {
            logger_type: setup_logger(logger_type.capitalize(), logger_configs[logger_type]['log_file'], logger_configs[logger_type]['config'])
            for logger_type in logger_types
        }
        return loggers['training'], loggers['visualization']

    def _setup_model(self, vocab_size, total_foundational_steps=1, warmup_steps=100):
        """Instantiates the model, loss function, optimizer, and scheduler."""
        self.logger.info("Setting up model, optimizer, criterion, and scheduler...")
        self.model = TidalLanguageModel(vocab_size=vocab_size, config=self.config, experiment_dir=self.exp_dir)
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss()
        min_lr = self.config.get("LEARNING_RATE_SCHEDULER", {}).get("MIN_LR", 1e-6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=min_lr)
        self.scheduler = DynamicLRScheduler(
            self.optimizer, self.config, total_foundational_steps, warmup_steps 
        )

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
        desired_batch_size = self.config.get("DESIRED_BATCH_SIZE", self.config["BATCH_SIZE"])
        micro_batch_size = self.config["BATCH_SIZE"]
        if desired_batch_size % micro_batch_size != 0:
            raise ValueError("DESIRED_BATCH_SIZE must be divisible by BATCH_SIZE for gradient accumulation.")
        accumulation_steps = desired_batch_size // micro_batch_size

        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch_num} Training")

        self.optimizer.zero_grad() # Zero gradients at the start of the epoch

        for i, (input_sequence, target_sequence) in enumerate(progress_bar):
            input_sequence_gpu = input_sequence.to(self.device)
            target_sequence_gpu = target_sequence.to(self.device)
            
            with autocast(self.device):
                logits, physics_loss, viz_data = self.model(input_sequence_gpu, target_sequence_gpu)
                final_physics_loss, physics_loss_components = physics_loss

                vocab_size = logits.shape[-1]
                prediction_loss = self.criterion(logits.view(-1, vocab_size), target_sequence_gpu.view(-1))

                mean_physics_loss = final_physics_loss.mean()
                scalar_physics_loss = mean_physics_loss.item()
                loss = prediction_loss + self.config["PHYSICS_LOSS_WEIGHT"] * mean_physics_loss

                # Normalize loss for accumulation
                loss = loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["MAX_GRAD_NORM"])

                # Scaler steps the optimizer
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler:
                    self.scheduler.step(tide_name=tide_name)

                self.optimizer.zero_grad() # Reset gradients after update

            total_loss += loss.item() * accumulation_steps # Un-normalize for logging
            global_step = self.current_epoch_num * len(data_loader) + i

            log_data_dict = {
                'total_loss': loss.item() * accumulation_steps, 
                'prediction_loss': prediction_loss.item(), 
                'physics_loss': scalar_physics_loss,
                'physics_loss_components': {
                    'F_pos': physics_loss_components['F_pos'].item(),
                    'F_neg': physics_loss_components['F_neg'].item()
                },
                'viz_data': viz_data,
                'input_sequence': input_sequence.cpu(),
                'input_sequence_shape': input_sequence.shape[1],
                'vocab': vocab,
                'tide_name': tide_name
            }

            self._log_metrics(log_data_dict, global_step)

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

    def _tidal_fine_tuning_loop(self, vocab, phase_name, max_epochs, start_epoch=0, cache_freq=None):
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

    def run(self):
        """Main function to orchestrate the model training pipeline."""
        try:  
            self.logger.info("Building vocabulary...")
            vocab = load_vocab(self.config)

            # Create and static cache the reverse mapping.
            self.idx_to_word = {v: k for k, v in vocab.items()}

            self.probe_words = self.config.get("PROBE_WORDS", [])
            vocab_size = len(vocab)

            # This is an estimate, but it's crucial for the cosine annealing schedule.
            max_foundational_epochs = self.config["NUM_EPOCHS"]
            # Use a dummy dataset to find the number of batches.
            temp_dataset = load_or_create_dataset(self.config["FOUNDATIONAL_CORPUS_PATH"], self.config)
            desired_batch_size = self.config.get("DESIRED_BATCH_SIZE", self.config["BATCH_SIZE"])
            micro_batch_size = self.config["BATCH_SIZE"]
            accumulation_steps = desired_batch_size // micro_batch_size
            num_micro_batches_per_epoch = len(temp_dataset) // micro_batch_size
            num_optimizer_steps_per_epoch = num_micro_batches_per_epoch // accumulation_steps
            total_foundational_steps = max_foundational_epochs * num_optimizer_steps_per_epoch
            # Calculate warmup steps as a ratio of the total foundational steps
            warmup_ratio = self.config["LEARNING_RATE_SCHEDULER"].get("WARMUP_RATIO", 0.1) # Default to 10%
            warmup_steps = int(warmup_ratio * total_foundational_steps)
            self.logger.info(f"Total foundational steps: {total_foundational_steps}, Calculated warm-up steps: {warmup_steps}")

            self._setup_model(vocab_size, total_foundational_steps, warmup_steps)

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
                self._training_loop(loader, foundational_phase_name, max_foundational_epochs, start_epoch_foundational, cache_freq=self.config.get("TRAINING_MODEL_ARTIFACT_CACHE_FREQUENCY", 1), cache_milestones=[max_foundational_epochs // 2], vocab=vocab)

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
                        self._training_loop(loader, early_tune_phase_name, early_tune_epochs, start_epoch_early, cache_freq=self.config.get("TRAINING_MODEL_ARTIFACT_CACHE_FREQUENCY", 1), vocab=vocab)
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
                self._tidal_fine_tuning_loop(vocab, tidal_tune_phase_name, max_finetune_epochs, start_epoch_tidal, cache_freq=self.config.get("FINE_TUNE_MODEL_ARTIFACT_CACHE_FREQUENCY", 1))

            final_model_path = os.path.join(self.exp_dir, f"{self.config['TIDAL_MODEL_NAME']}_v{self.config['TIDAL_MODEL_VERSION']}.pth")

            self.logger.info(f"\n--- Training complete. Saving final model to {final_model_path} ---")
            torch.save(self.model.state_dict(), final_model_path)

            artifact = wandb.Artifact(
                name=f"{self.config['TIDAL_MODEL_NAME']}-v{self.config['TIDAL_MODEL_VERSION']}", 
                type='model'
            )
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)
            final_embeddings_512d = self.model.position_embeddings.weight.detach()
            final_embeddings_8d = self.model.projection_layer(final_embeddings_512d).cpu().numpy()
            idx_to_word = {v: k for k, v in vocab.items()}
            metadata = [idx_to_word.get(i, '<UNK>') for i in range(len(vocab))]
            columns = ["id", "word"] + [f"dim_{i}" for i in range(final_embeddings_8d.shape[1])]

            table_data = []
            for i, word in enumerate(metadata):
                row = [i, word] + list(final_embeddings_8d[i])
                table_data.append(row)

            embedding_table = wandb.Table(data=table_data, columns=columns)
            wandb.log({"Semantic_Space_8D_Embeddings": embedding_table})
        
            return final_model_path

        finally:
            self.logger.info("Shutting down visualization executor...")
            self.visualization_executor.shutdown()
