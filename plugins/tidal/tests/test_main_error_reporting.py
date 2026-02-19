"""Tests that Main.py reports errors to stderr and exits non-zero."""

import subprocess
import sys
import unittest

from plugins.tidal.tests.timeout import TimedTestCase


class TestMainErrorReporting(TimedTestCase):
    """Main.py must write tracebacks to stderr and exit non-zero on failure."""

    def _run_main(self, extra_args=None):
        """Run Main.py as a subprocess and capture outputs."""
        cmd = [
            sys.executable, "-m", "plugins.tidal.Main",
            "--config", "nonexistent_config_that_does_not_exist.yaml",
        ]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result

    def test_exits_nonzero_on_missing_config(self):
        """Main.py should exit with code 1 when config file is missing."""
        result = self._run_main()
        self.assertNotEqual(result.returncode, 0, "Expected non-zero exit code")

    def test_writes_error_to_stderr_on_training_failure(self):
        """When training raises, the traceback must appear on stderr.

        We use a config file that exists but will cause a downstream error
        to test the except Exception handler, not the config-not-found path.
        """
        # Create a minimal config that will fail during Trainer init
        import os
        import tempfile

        from ruamel.yaml import YAML
        yaml = YAML()

        config = {
            "VOCAB_SIZE": 100,
            "EMBED_DIM": 32,
            "NUM_HEADS": 2,
            "FFN_DIM": 64,
            "NUM_LAYERS": 1,
            "MAX_SEQ_LENGTH": 16,
            "BATCH_SIZE": 2,
            "GRAD_ACCUM_STEPS": 1,
            "LEARNING_RATE": 0.001,
            "EPOCHS": 1,
            "ENABLE_CONSOLE_LOGGING": False,
            # Missing or bad dataset config to trigger an error
            "DATASET_NAME": "nonexistent_dataset_that_will_fail_12345",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "plugins.tidal.Main", "--config", config_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Must exit non-zero
            self.assertNotEqual(
                result.returncode, 0,
                f"Expected non-zero exit code, got {result.returncode}.\n"
                f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}",
            )

            # Must have error info on stderr
            self.assertTrue(
                len(result.stderr.strip()) > 0,
                "Expected error output on stderr, got nothing.\n"
                f"stdout: {result.stdout[:500]}",
            )
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
