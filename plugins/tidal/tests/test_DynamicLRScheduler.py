import math
import unittest

import torch
import torch.nn as nn

from plugins.tidal.DynamicLRScheduler import DynamicLRScheduler


class TestDynamicLRScheduler(unittest.TestCase):
    """Unit tests for the DynamicLRScheduler learning rate schedule."""

    def _make_scheduler(self, total_steps, base_lr=0.001, min_lr=1e-6, warmup_ratio=0.1):
        """Helper to create a scheduler with a dummy optimizer."""
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        config = {
            "LEARNING_RATE_SCHEDULER": {
                "BASE_LR": base_lr,
                "MIN_LR": min_lr,
                "WARMUP_RATIO": warmup_ratio,
            }
        }
        scheduler = DynamicLRScheduler(optimizer, config, total_steps)
        return scheduler, optimizer

    def test_lr_reaches_base_after_warmup(self):
        """After warmup completes, LR should be approximately BASE_LR."""
        total_steps = 1000
        base_lr = 0.001
        scheduler, optimizer = self._make_scheduler(total_steps, base_lr=base_lr)

        warmup_steps = int(total_steps * 0.1)  # 100 steps
        for _ in range(warmup_steps):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, base_lr, places=5,
                               msg=f"LR should be ~{base_lr} after warmup, got {lr}")

    def test_lr_reaches_min_after_all_steps(self):
        """After all steps complete, LR should be approximately MIN_LR."""
        total_steps = 1000
        min_lr = 1e-6
        scheduler, optimizer = self._make_scheduler(total_steps, min_lr=min_lr)

        for _ in range(total_steps):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, min_lr, places=8,
                               msg=f"LR should be ~{min_lr} after all steps, got {lr}")

    def test_cosine_annealing_midpoint(self):
        """At the midpoint of cosine phase, LR should be midway between BASE_LR and MIN_LR."""
        total_steps = 1000
        base_lr = 0.001
        min_lr = 1e-6
        warmup_ratio = 0.1
        scheduler, optimizer = self._make_scheduler(
            total_steps, base_lr=base_lr, min_lr=min_lr, warmup_ratio=warmup_ratio,
        )

        warmup_steps = int(total_steps * warmup_ratio)  # 100
        cosine_steps = total_steps - warmup_steps  # 900
        midpoint_step = warmup_steps + cosine_steps // 2  # 550

        for _ in range(midpoint_step):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        expected_mid = (base_lr + min_lr) / 2.0
        self.assertAlmostEqual(lr, expected_mid, places=5,
                               msg=f"LR at cosine midpoint should be ~{expected_mid}, got {lr}")

    def test_total_steps_matches_optimizer_steps(self):
        """Simulate Trainer step counting with gradient accumulation.

        With accumulation_steps=32, the scheduler should receive
        total_steps = num_epochs * num_batches // accumulation_steps,
        and after running all epochs it should reach MIN_LR.
        """
        num_epochs = 12
        num_batches = 6400  # representative micro-batch count per epoch
        accumulation_steps = 32
        base_lr = 0.001
        min_lr = 1e-6

        # This is the corrected calculation (optimizer steps, not micro-batches)
        total_optimizer_steps = num_epochs * num_batches // accumulation_steps

        scheduler, optimizer = self._make_scheduler(
            total_optimizer_steps, base_lr=base_lr, min_lr=min_lr,
        )

        # Simulate training: step() is called once per accumulation cycle
        for epoch in range(num_epochs):
            for micro_batch in range(num_batches):
                if (micro_batch + 1) % accumulation_steps == 0:
                    scheduler.step()

        self.assertEqual(scheduler.current_step, total_optimizer_steps,
                         "Scheduler current_step should equal total optimizer steps")

        lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, min_lr, places=8,
                               msg=f"LR should be ~{min_lr} after all optimizer steps, got {lr}")


if __name__ == "__main__":
    unittest.main()
