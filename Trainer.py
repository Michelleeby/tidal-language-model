import torch
import torch.optim as optim
import torch.nn as nn
import os
import re
import glob

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DataPipeline import TinyStoriesDataset, get_tokenizer
from Utils import setup_logger
from DynamicLRScheduler import DynamicLRScheduler
from MetricsLogger import MetricsLogger


class Trainer:
    """
    Training pipeline for the TransformerLM.

    Phase 1: Standard cross-entropy pretraining on TinyStories with neutral gate signals.
    Phase 2 (optional): RL-controlled gate training (delegates to RLTrainer).
    """

    def __init__(self, config, experiment_dir):
        self.config = config
        self.exp_dir = experiment_dir

        self.device = self._get_device()
        self.logger = self._setup_logger(config.copy())

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler()
        self.current_epoch_num = 0

        self.tags = self.config.get("TENSORBOARD_TAGS", {})

        tensorboard_dir = os.path.join(self.exp_dir, "tensorboard_logs")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        self.metrics_logger = MetricsLogger(self.exp_dir)

    def _log_metrics(self, data, global_step):
        """Log metrics to TensorBoard and the dashboard MetricsLogger."""
        metrics_dict = {
            "Losses/Total": data["total_loss"],
            "Losses/Prediction": data["prediction_loss"],
            "Learning Rate": self.optimizer.param_groups[0]["lr"],
        }

        self.tb_writer.add_scalar("Losses/Total", data["total_loss"], global_step)
        self.tb_writer.add_scalar("Losses/Prediction", data["prediction_loss"], global_step)
        self.tb_writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], global_step)

        self.metrics_logger.log_metrics(metrics_dict, global_step)

    def _get_device(self):
        if self.config["DEVICE"] == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config["DEVICE"]

    def _setup_logger(self, config):
        config["ENABLE_CONSOLE_LOGGING"] = True
        return setup_logger(
            "Training",
            os.path.join(self.exp_dir, "training.log"),
            config,
        )

    def _setup_model(self, vocab_size, total_foundational_steps=1):
        """Instantiates the model, loss function, optimizer, and scheduler."""
        from TransformerLM import TransformerLM

        self.logger.info("Setting up TransformerLM, optimizer, criterion, and scheduler...")

        self.model = TransformerLM(vocab_size=vocab_size, config=self.config)
        self.model.to(self.device)
        self.model = torch.compile(self.model)

        self.criterion = nn.CrossEntropyLoss()
        base_lr = self.config.get("LEARNING_RATE_SCHEDULER", {}).get("BASE_LR", 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
        self.scheduler = DynamicLRScheduler(self.optimizer, self.config, total_foundational_steps)

    def _get_data_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.config.get("NUM_CPU_CORE_WORKERS", 4),
            pin_memory=True,
            persistent_workers=True,
        )

    def _train_epoch(self, data_loader):
        """Train for one epoch with neutral gate signals."""
        self.model.train()
        desired_batch_size = self.config.get("DESIRED_BATCH_SIZE", self.config["BATCH_SIZE"])
        micro_batch_size = self.config["BATCH_SIZE"]
        if desired_batch_size % micro_batch_size != 0:
            raise ValueError("DESIRED_BATCH_SIZE must be divisible by BATCH_SIZE for gradient accumulation.")
        accumulation_steps = desired_batch_size // micro_batch_size

        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch_num} Training")

        self.optimizer.zero_grad()

        for i, (input_sequence, target_sequence) in enumerate(progress_bar):
            input_sequence_gpu = input_sequence.to(self.device)
            target_sequence_gpu = target_sequence.to(self.device)

            with autocast(self.device):
                logits, (prediction_loss, _), _ = self.model(
                    input_sequence_gpu, target_sequence_gpu,
                )
                loss = prediction_loss / accumulation_steps

            self.scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["MAX_GRAD_NORM"])

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            global_step = self.current_epoch_num * len(data_loader) + i

            log_data_dict = {
                "total_loss": loss.item() * accumulation_steps,
                "prediction_loss": prediction_loss.item(),
            }
            self._log_metrics(log_data_dict, global_step)

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        self.current_epoch_num += 1
        return avg_loss

    def _save_checkpoint(self, epoch, phase_name):
        phase_name_slug = phase_name.lower().replace("-", "_")
        checkpoint_name = f"checkpoint_{phase_name_slug}_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.exp_dir, checkpoint_name)
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        state_dict = self.model._orig_mod.state_dict() if hasattr(self.model, "_orig_mod") else self.model.state_dict()
        torch.save(state_dict, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            model_to_load = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            model_to_load.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.logger.info("Successfully loaded checkpoint.")
        except FileNotFoundError:
            self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def _find_latest_checkpoint(self, phase_name):
        phase_name_slug = phase_name.lower().replace("-", "_")
        search_pattern = os.path.join(self.exp_dir, f"checkpoint_{phase_name_slug}_epoch_*.pth")
        checkpoint_files = glob.glob(search_pattern)

        if not checkpoint_files:
            return None, 0

        latest_epoch = -1
        latest_checkpoint = None

        for f_path in checkpoint_files:
            basename = os.path.basename(f_path)
            match = re.search(r"epoch_(\d+).pth", basename)
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
        """Standard training loop with early stopping."""
        self.scheduler.set_phase("foundational")
        self.logger.info(f"--- Starting {phase_name} Phase (resuming from epoch {start_epoch}) ---")
        best_loss = float("inf")
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

    def run(self):
        """Main function to orchestrate the model training pipeline."""
        try:
            tokenizer = get_tokenizer()
            vocab_size = self.config.get("VOCAB_SIZE", tokenizer.vocab_size)
            max_length = self.config.get("MAX_CONTEXT_LENGTH", 256)

            self.logger.info("Loading TinyStories training data...")
            train_dataset = TinyStoriesDataset(
                split="train",
                max_length=max_length,
                tokenizer=tokenizer,
            )

            max_foundational_epochs = self.config["NUM_EPOCHS"]
            num_batches = len(train_dataset) // self.config["BATCH_SIZE"]
            total_foundational_steps = max_foundational_epochs * num_batches

            self._setup_model(vocab_size, total_foundational_steps)

            # Phase 1: Foundational Pre-training (neutral gate signals)
            foundational_phase_name = "Foundational"
            checkpoint_path, start_epoch_foundational = self._find_latest_checkpoint(foundational_phase_name)
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)

            if start_epoch_foundational >= max_foundational_epochs:
                self.logger.info("Foundational training already completed. Skipping.")
            else:
                loader = self._get_data_loader(train_dataset)
                self._training_loop(
                    loader,
                    foundational_phase_name,
                    max_foundational_epochs,
                    start_epoch_foundational,
                    cache_freq=self.config.get("TRAINING_MODEL_ARTIFACT_CACHE_FREQUENCY", 1),
                    cache_milestones=[max_foundational_epochs // 2],
                )

            # Save final model
            model_name = self.config.get("MODEL_NAME", "transformer-lm")
            model_version = self.config.get("MODEL_VERSION", "1.0.0")
            final_model_path = os.path.join(self.exp_dir, f"{model_name}_v{model_version}.pth")

            self.logger.info(f"\n--- Training complete. Saving final model to {final_model_path} ---")
            state_dict = self.model._orig_mod.state_dict() if hasattr(self.model, "_orig_mod") else self.model.state_dict()
            torch.save(state_dict, final_model_path)

            self.metrics_logger.finalize()

            return final_model_path

        finally:
            self.tb_writer.close()
