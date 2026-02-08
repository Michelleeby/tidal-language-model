import unittest
import torch
import os

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from ConstantLanguageModel import ConstantLanguageModel

class TestConstantLanguageModel(unittest.TestCase):
    """
    Unit tests for the ConstantLanguageModel class.
    These tests verify the model's initialization, forward pass output shapes,
    generation capabilities, and compatibility with the training infrastructure.
    """

    @classmethod
    def setUpClass(cls):
        """Load the configuration from the YAML file once for all tests."""
        try:
            with open(os.path.join('configs', 'constant_base_config.yaml'), 'r') as f:
                cls.config = yaml.load(f)
        except FileNotFoundError:
            raise Exception("constant_base_config.yaml not found. Make sure it's in the configs/ directory.")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing constant_base_config.yaml: {e}")

    def setUp(self):
        """Set up a ConstantLanguageModel instance for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 100  # A small vocab size for testing
        # Ensure log directory exists for the test logger
        os.makedirs(self.config.get("LOG_DIRECTORY", "logs"), exist_ok=True)
        self.model = ConstantLanguageModel(vocab_size=self.vocab_size, config=self.config, experiment_dir="logs")
        self.model.to(self.device)

    def test_initialization(self):
        """Test if the model and its components are initialized correctly."""
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.embed_dim, self.config.get("EMBED_DIM", 512))
        self.assertEqual(self.model.num_transformer_blocks, self.config.get("NUM_TRANSFORMER_BLOCKS", 6))

        # Check embedding dimensions
        self.assertEqual(self.model.token_embeddings.weight.shape, (self.vocab_size, self.config["EMBED_DIM"]))
        self.assertEqual(self.model.position_embeddings.weight.shape, (self.config["MAX_CONTEXT_LENGTH"], self.config["EMBED_DIM"]))

        # Check output projection
        self.assertEqual(self.model.output_projection.weight.shape, (self.vocab_size, self.config["EMBED_DIM"]))

    def test_forward_pass_output_shapes(self):
        """Test the shapes of tensors returned by the forward pass."""
        batch_size = 4
        seq_len = 20
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        logits, (total_loss, loss_components), viz_data = self.model(input_ids, target_ids)

        # Check logits shape
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

        # Check loss
        self.assertTrue(torch.is_tensor(total_loss))
        self.assertEqual(total_loss.numel(), 1)

        # Check loss components (should be None for constant model)
        self.assertIsNone(loss_components)

        # Check visualization data (should be empty dict for constant model)
        self.assertIsInstance(viz_data, dict)
        self.assertEqual(len(viz_data), 0)

    def test_forward_pass_without_targets(self):
        """Test forward pass without targets (inference mode)."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, (total_loss, _), _ = self.model(input_ids, target_ids=None)

        # Check logits shape
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

        # Loss should be zero when no targets provided
        self.assertEqual(total_loss.item(), 0.0)

    def test_generate_method(self):
        """Test the autoregressive generate method."""
        prompt_ids = torch.tensor([1, 5, 10], device=self.device)
        max_new_tokens = 5

        generated_ids = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=10
        )

        # Check that generation returns a list
        self.assertIsInstance(generated_ids, list)

        # Check that it includes prompt + new tokens
        self.assertEqual(len(generated_ids), len(prompt_ids) + max_new_tokens)

        # Check that all generated IDs are valid vocab indices
        for token_id in generated_ids:
            self.assertTrue(0 <= token_id < self.vocab_size)

    def test_generate_with_temperature(self):
        """Test generation with different temperature settings."""
        prompt_ids = torch.tensor([1, 5, 10], device=self.device)
        max_new_tokens = 3

        # Test with low temperature (more deterministic)
        gen_low_temp = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_k=10
        )

        # Test with high temperature (more random)
        gen_high_temp = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=2.0,
            top_k=10
        )

        # Both should generate valid outputs
        self.assertEqual(len(gen_low_temp), len(prompt_ids) + max_new_tokens)
        self.assertEqual(len(gen_high_temp), len(prompt_ids) + max_new_tokens)

    def test_repetition_penalty(self):
        """Test generation with repetition penalty."""
        prompt_ids = torch.tensor([1, 2, 3], device=self.device)
        max_new_tokens = 10

        generated_ids = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=self.vocab_size,
            repetition_penalty=2.0  # High penalty to discourage repetition
        )

        # Check that generation completes successfully
        self.assertEqual(len(generated_ids), len(prompt_ids) + max_new_tokens)

    def test_no_physics_simulator_attribute(self):
        """Test that constant model doesn't have physics simulator."""
        self.assertFalse(hasattr(self.model, 'physics_simulator'))

    def test_no_endocrine_system_attribute(self):
        """Test that constant model doesn't have endocrine system."""
        self.assertFalse(hasattr(self.model, 'endocrine_system'))
        self.assertFalse(hasattr(self.model, 'enable_endocrine_system'))

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        self.model.train()
        logits, (loss, _), _ = self.model(input_ids, target_ids)

        # Perform backward pass
        loss.backward()

        # Check that gradients exist for key parameters
        self.assertIsNotNone(self.model.token_embeddings.weight.grad)
        self.assertIsNotNone(self.model.output_projection.weight.grad)

        # Check that gradients are non-zero (at least for some parameters)
        has_nonzero_grad = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        self.assertTrue(has_nonzero_grad)

    def test_config_model_type(self):
        """Test that config specifies constant model type."""
        self.assertEqual(self.config.get("MODEL_TYPE"), "constant")

    def test_interface_compatibility_with_tidal_model(self):
        """Test that constant model has the same interface as Tidal model."""
        # Check that generate method exists with same signature
        self.assertTrue(hasattr(self.model, 'generate'))

        # Check that forward returns same structure
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        output = self.model(input_ids, target_ids)

        # Should return 3-tuple: (logits, loss_tuple, viz_data)
        self.assertEqual(len(output), 3)

        logits, loss_tuple, viz_data = output

        # loss_tuple should be 2-tuple: (total_loss, loss_components)
        self.assertEqual(len(loss_tuple), 2)

if __name__ == '__main__':
    unittest.main()
