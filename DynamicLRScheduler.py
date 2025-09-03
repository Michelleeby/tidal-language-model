import math

class DynamicLRScheduler:
    """
    Manages the learning rate with distinct phases: warm-up, cosine annealing, and fine-tuning.
    """
    def __init__(self, optimizer, config: dict, total_foundational_steps: int):
        self.optimizer = optimizer
        self.config = config["LEARNING_RATE_SCHEDULER"]
        
        # Scheduler parameters
        self.warmup_steps = self.config["WARMUP_STEPS"]
        self.base_lr = self.config["BASE_LR"]
        self.min_lr = self.config["MIN_LR"]
        self.total_foundational_steps = total_foundational_steps

        # State management
        self.current_step = 0
        self.phase = "warmup" # Initial phase

    def set_phase(self, phase: str):
        """Sets the current training phase ('warmup', 'foundational', or 'finetuning')."""
        self.phase = phase
        print(f"LR Scheduler phase changed to: {self.phase.upper()}")

    def _get_lr_for_step(self, tide_name: str = None):
        """Calculates the learning rate based on the current phase and step."""
        if self.phase == "finetuning":
            # Phase 3: Tide-specific constant LR
            if tide_name == "high":
                return self.config["FINETUNE_LR_HIGH"]
            elif tide_name == "low":
                return self.config["FINETUNE_LR_LOW"]
            elif tide_name == "storm":
                return self.config["FINETUNE_LR_STORM"]
            else:
                return self.config["FINETUNE_LR_LOW"] # Default for safety

        # Phase 1: Linear Warm-up
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)

        # Phase 2: Cosine Annealing Decay
        progress = (self.current_step - self.warmup_steps) / (self.total_foundational_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def step(self, tide_name: str = None):
        """
        Updates the optimizer's learning rate and increments the step counter.
        Should be called after each optimizer.step().
        """
        self.current_step += 1
        lr = self._get_lr_for_step(tide_name)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr