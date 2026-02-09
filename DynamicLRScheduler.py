import math


class DynamicLRScheduler:
    """
    Manages learning rate with two phases: warmup and cosine annealing.
    """

    def __init__(self, optimizer, config: dict, total_foundational_steps: int):
        self.optimizer = optimizer
        self.config = config["LEARNING_RATE_SCHEDULER"]

        if "WARMUP_STEPS" in self.config:
            self.warmup_steps = self.config["WARMUP_STEPS"]
        else:
            warmup_ratio = self.config.get("WARMUP_RATIO", 0.1)
            self.warmup_steps = int(total_foundational_steps * warmup_ratio)

        self.base_lr = self.config["BASE_LR"]
        self.min_lr = self.config["MIN_LR"]
        self.total_foundational_steps = total_foundational_steps

        self.current_step = 0
        self.phase = "warmup"

    def set_phase(self, phase: str):
        """Sets the current training phase ('warmup' or 'foundational')."""
        self.phase = phase
        print(f"LR Scheduler phase changed to: {self.phase.upper()}")

    def _get_lr_for_step(self):
        # Phase 1: Linear Warm-up
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)

        # Phase 2: Cosine Annealing Decay
        progress = (self.current_step - self.warmup_steps) / max(self.total_foundational_steps - self.warmup_steps, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def step(self):
        """Updates the optimizer's learning rate and increments the step counter."""
        self.current_step += 1
        lr = self._get_lr_for_step()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
