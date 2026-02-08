import torch
import torch.optim as optim
import torch.nn as nn
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SequentialDataset import load_or_create_dataset, load_vocab
from Utils import setup_logger, plot_semantic_space, format_cluster_analysis_text
from DynamicLRScheduler import DynamicLRScheduler
from MetricsLogger import MetricsLogger

class Trainer:
    """
    Handles the entire training pipeline for the Tidal Language Model and
    the Constant Language Model (baseline), including early stopping,
    checkpointing, and resuming from checkpoints.
    """
    def __init__(self, config, experiment_dir):
        self.config = config
        self.exp_dir = experiment_dir
        self.model_type = config.get("MODEL_TYPE", "tidal")

        self.device = self._get_device()
        self.logger, self.logger_visualization = self._setup_loggers(config.copy())
        self.last_viz_time = 0
        self.qualitative_log_buffer = []
        self.pending_viz_futures = []

        # Initialize a thread pool with one worker for async I/O operations.
        # This prevents plotting and file saving from blocking the main training thread.
        self.visualization_executor = ThreadPoolExecutor(max_workers=1)

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler()
        self.current_epoch_num = 0

        self.tags = self.config.get("TENSORBOARD_TAGS", {})

        # TensorBoard writer
        tensorboard_dir = os.path.join(self.exp_dir, "tensorboard_logs")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        # Dashboard metrics logger
        self.metrics_logger = MetricsLogger(self.exp_dir)

    def _log_qualitative_events_async(self, log_entries, idx_to_word, global_step):
        """Processes a batch of trigger events and logs them to MetricsLogger."""
        if not log_entries:
            return

        try:
            for entry in log_entries:
                tokens = [idx_to_word.get(idx, "<UNK>") for idx in entry["token_ids"]]
                text = " ".join(tokens)
                strength = entry["strength"]
                threshold = entry["threshold"]

                event_data = {
                    "trigger_name": entry["trigger_name"],
                    "hormone_name": entry["hormone_name"],
                    "strength": strength,
                    "threshold": threshold,
                    "tidal_level": entry["tidal_level"],
                    "active": strength > threshold,
                    "text": text
                }

                self.metrics_logger.log_trigger_event(event_data, global_step)

        except Exception as e:
            self.logger_visualization.error(f"Error during async qualitative logging: {e}", exc_info=True)

    def _log_visuals_async(self, viz_data, global_step, batch_tokens, seq_len, vocab, tide_name):
        """Logs visualizations in a separate thread to avoid blocking training."""
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

            # Log to TensorBoard as image
            self.tb_writer.add_figure("Semantic Space", fig, global_step=global_step)

            # Save semantic space data for the dashboard
            semantic_data = {
                'positions_2d': viz_data_sliced['positions_2d'].cpu().numpy(),
                'positions_8d': viz_data_sliced['positions_8d'].cpu().numpy(),
                'forces_2d': viz_data_sliced['forces_2d'].cpu().numpy(),
                'masses': viz_data_sliced['masses'].cpu().numpy(),
                'token_ids': token_ids_for_viz.cpu().tolist()
            }
            self.metrics_logger.log_semantic_space(semantic_data, global_step)

            # Save frame to disk for video compilation
            if self.config.get("ENABLE_VISUALIZATION", True):
                frames_dir = os.path.join(self.exp_dir, self.config.get("TRAINING_FRAMES_SUBDIR", "semantic_space_frames"))
                os.makedirs(frames_dir, exist_ok=True)
                frame_path = os.path.join(frames_dir, f"frame_{global_step:08d}.png")
                fig.savefig(frame_path, dpi=100, bbox_inches='tight')

        except Exception as e:
            self.logger_visualization.error(f"Error during async visualization: {e}", exc_info=True)
        finally:
            if 'fig' in locals() and fig:
                plt.close(fig)

    def _log_metrics(self, data, global_step):
        """Log metrics to TensorBoard and the dashboard MetricsLogger."""
        is_tidal = self.model_type == "tidal"

        # Build the metrics dict for dashboard logging
        metrics_dict = {
            "Losses/Total": data['total_loss'],
            "Losses/Prediction": data['prediction_loss'],
            "Learning Rate": self.optimizer.param_groups[0]['lr']
        }

        # TensorBoard scalar logging
        self.tb_writer.add_scalar("Losses/Total", data['total_loss'], global_step)
        self.tb_writer.add_scalar("Losses/Prediction", data['prediction_loss'], global_step)
        self.tb_writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], global_step)

        if is_tidal:
            metrics_dict["Losses/Physics"] = data['physics_loss']
            metrics_dict["Energy/F_pos"] = data['physics_loss_components']['F_pos']
            metrics_dict["Energy/F_neg"] = data['physics_loss_components']['F_neg']
            metrics_dict["Physics/G"] = self.model.physics_simulator.G.item()
            metrics_dict["Physics/Repulsion_Strength"] = self.model.physics_simulator.repulsion_strength.item()
            metrics_dict["Physics/Well_Attraction"] = self.model.physics_simulator.well_attraction_strength.item()
            metrics_dict["Physics/Temperature"] = self.model.physics_simulator.temperature.item()

            self.tb_writer.add_scalar("Losses/Physics", data['physics_loss'], global_step)
            self.tb_writer.add_scalar("Energy/F_pos", data['physics_loss_components']['F_pos'], global_step)
            self.tb_writer.add_scalar("Energy/F_neg", data['physics_loss_components']['F_neg'], global_step)
            self.tb_writer.add_scalar("Physics/G", self.model.physics_simulator.G.item(), global_step)
            self.tb_writer.add_scalar("Physics/Repulsion_Strength", self.model.physics_simulator.repulsion_strength.item(), global_step)
            self.tb_writer.add_scalar("Physics/Well_Attraction", self.model.physics_simulator.well_attraction_strength.item(), global_step)
            self.tb_writer.add_scalar("Physics/Temperature", self.model.physics_simulator.temperature.item(), global_step)

            if self.model.enable_endocrine_system:
                hormone_state = self.model.endocrine_system.get_hormone_state()
                for k, v in hormone_state.items():
                    metrics_dict[f"Hormones/{k}"] = v
                    self.tb_writer.add_scalar(f"Hormones/{k}", v, global_step)

        # Log to dashboard JSONL
        self.metrics_logger.log_metrics(metrics_dict, global_step)

        # Schedule async visualization (tidal model only, with viz data)
        if is_tidal and 'viz_data' in data and data['viz_data']:
            current_time = time.time()
            viz_interval_seconds = self.config.get("VISUALIZATION_INTERVAL_SECONDS", 60)
            if current_time - self.last_viz_time > viz_interval_seconds:
                self.last_viz_time = current_time

                # Offload slow I/O to the background thread.
                # Pass all data as CPU tensors to avoid GPU context issues.
                future = self.visualization_executor.submit(
                    self._log_visuals_async,
                    data['viz_data'],
                    global_step,
                    data['input_sequence'],
                    data['input_sequence_shape'],
                    data['vocab'],
                    data['tide_name']
                )
                self.pending_viz_futures.append(future)

        # Schedule async qualitative logging (tidal model only)
        if is_tidal:
            qualitative_log_interval = self.config.get("QUALITATIVE_LOG_INTERVAL_BATCHES", 100)
            if global_step % qualitative_log_interval == 0 and self.qualitative_log_buffer:
                logs_to_process = self.qualitative_log_buffer.copy()
                self.qualitative_log_buffer.clear()

                future = self.visualization_executor.submit(
                    self._log_qualitative_events_async,
                    logs_to_process,
                    self.idx_to_word,
                    global_step
                )
                self.pending_viz_futures.append(future)

        # Drain completed async futures
        still_pending = []
        for future in self.pending_viz_futures:
            if not future.done():
                still_pending.append(future)
                continue
            try:
                future.result()
            except Exception as e:
                self.logger_visualization.error(f"Async logging task failed: {e}", exc_info=True)
        self.pending_viz_futures = still_pending

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

    def _setup_model(self, vocab_size, total_foundational_steps=1):
        """Instantiates the model, loss function, optimizer, and scheduler."""
        self.logger.info(f"Setting up {self.model_type} model, optimizer, criterion, and scheduler...")

        if self.model_type == "constant":
            from ConstantLanguageModel import ConstantLanguageModel
            self.model = ConstantLanguageModel(vocab_size=vocab_size, config=self.config, experiment_dir=self.exp_dir)
        else:
            from TidalLanguageModel import TidalLanguageModel
            self.model = TidalLanguageModel(vocab_size=vocab_size, config=self.config, experiment_dir=self.exp_dir)

        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss()
        base_lr = self.config.get("LEARNING_RATE_SCHEDULER", {}).get("BASE_LR", 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
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
        desired_batch_size = self.config.get("DESIRED_BATCH_SIZE", self.config["BATCH_SIZE"])
        micro_batch_size = self.config["BATCH_SIZE"]
        if desired_batch_size % micro_batch_size != 0:
            raise ValueError("DESIRED_BATCH_SIZE must be divisible by BATCH_SIZE for gradient accumulation.")
        accumulation_steps = desired_batch_size // micro_batch_size

        is_tidal = self.model_type == "tidal"
        physics_loss_weight = self.config.get("PHYSICS_LOSS_WEIGHT", 0.0) if is_tidal else 0.0

        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch_num} Training")

        self.optimizer.zero_grad()

        for i, (input_sequence, target_sequence) in enumerate(progress_bar):
            input_sequence_gpu = input_sequence.to(self.device)
            target_sequence_gpu = target_sequence.to(self.device)

            with autocast(self.device):
                logits, physics_loss_tuple, viz_data = self.model(input_sequence_gpu, target_sequence_gpu)

                vocab_size = logits.shape[-1]
                prediction_loss = self.criterion(logits.view(-1, vocab_size), target_sequence_gpu.view(-1))

                if is_tidal:
                    final_physics_loss, physics_loss_components = physics_loss_tuple
                    mean_physics_loss = final_physics_loss.mean()
                    scalar_physics_loss = mean_physics_loss.item()
                    loss = prediction_loss + physics_loss_weight * mean_physics_loss
                else:
                    scalar_physics_loss = 0.0
                    physics_loss_components = None
                    loss = prediction_loss

                loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["MAX_GRAD_NORM"])

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler:
                    self.scheduler.step(tide_name=tide_name)

                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            global_step = self.current_epoch_num * len(data_loader) + i

            log_data_dict = {
                'total_loss': loss.item() * accumulation_steps,
                'prediction_loss': prediction_loss.item(),
                'physics_loss': scalar_physics_loss,
                'tide_name': tide_name
            }

            if is_tidal:
                log_data_dict['physics_loss_components'] = {
                    'F_pos': physics_loss_components['F_pos'].item(),
                    'F_neg': physics_loss_components['F_neg'].item()
                }
                log_data_dict['viz_data'] = viz_data
                log_data_dict['input_sequence'] = input_sequence.cpu()
                log_data_dict['input_sequence_shape'] = input_sequence.shape[1]
                log_data_dict['vocab'] = vocab

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

    def _save_final_embeddings(self, vocab):
        """Save final 8D semantic embeddings to CSV (tidal model only)."""
        model_ref = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        final_embeddings_512d = model_ref.position_embeddings.weight.detach()
        final_embeddings_8d = model_ref._project(final_embeddings_512d, "down_512_to_8").cpu().numpy()

        idx_to_word = {v: k for k, v in vocab.items()}
        csv_path = os.path.join(self.exp_dir, "final_embeddings_8d.csv")

        axis_names = ["G", "X", "V", "A", "H", "S", "F", "T"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "word"] + axis_names)
            for i in range(len(vocab)):
                word = idx_to_word.get(i, '<UNK>')
                row = [i, word] + [f"{v:.6f}" for v in final_embeddings_8d[i]]
                writer.writerow(row)

        self.logger.info(f"Saved final 8D embeddings to {csv_path}")

    def run(self):
        """Main function to orchestrate the model training pipeline."""
        try:
            self.logger.info("Building vocabulary...")
            vocab = load_vocab(self.config)

            self.idx_to_word = {v: k for k, v in vocab.items()}
            self.probe_words = self.config.get("PROBE_WORDS", [])
            vocab_size = len(vocab)

            max_foundational_epochs = self.config["NUM_EPOCHS"]
            temp_dataset = load_or_create_dataset(self.config["FOUNDATIONAL_CORPUS_PATH"], self.config)
            num_batches = len(temp_dataset) // self.config["BATCH_SIZE"]
            total_foundational_steps = max_foundational_epochs * num_batches

            self._setup_model(vocab_size, total_foundational_steps)

            is_tidal = self.model_type == "tidal"

            # Foundational Pre-training
            if is_tidal:
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

            # Fine-tuning (tidal model only)
            if is_tidal:
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

            # Save final model
            model_name = self.config.get('TIDAL_MODEL_NAME', self.config.get('MODEL_NAME', 'model'))
            model_version = self.config.get('TIDAL_MODEL_VERSION', self.config.get('MODEL_VERSION', '1.0.0'))
            final_model_path = os.path.join(self.exp_dir, f"{model_name}_v{model_version}.pth")

            self.logger.info(f"\n--- Training complete. Saving final model to {final_model_path} ---")
            state_dict = self.model._orig_mod.state_dict() if hasattr(self.model, '_orig_mod') else self.model.state_dict()
            torch.save(state_dict, final_model_path)

            # Save final embeddings (tidal model only)
            if is_tidal:
                self._save_final_embeddings(vocab)

            # Mark training as complete in dashboard
            self.metrics_logger.finalize()

            return final_model_path

        finally:
            self.logger.info("Shutting down visualization executor...")
            self.visualization_executor.shutdown()
            self.tb_writer.close()
