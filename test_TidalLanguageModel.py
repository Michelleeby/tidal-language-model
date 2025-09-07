import unittest
import torch
import os

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from TidalLanguageModel import TidalLanguageModel

class TestTidalLanguageModel(unittest.TestCase):
    """
    Unit tests for the TidalLanguageModel class.
    These tests verify the model's initialization, internal projection logic,
    forward pass output shapes, and generation capabilities.
    """

    @classmethod
    def setUpClass(cls):
        """Load the configuration from the YAML file once for all tests."""
        try:
            with open(os.path.join('configs', 'base_config.yaml'), 'r') as f:
                cls.config = yaml.load(f)
        except FileNotFoundError:
            raise Exception("base_config.yaml not found. Make sure it's in the root directory.")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing base_config.yaml: {e}")

    def setUp(self):
        """Set up a TidalLanguageModel instance for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 100 # A small vocab size for testing
        # Ensure log directory exists for the test logger
        os.makedirs(self.config.get("LOG_DIRECTORY", "logs"), exist_ok=True)
        self.model = TidalLanguageModel(vocab_size=self.vocab_size, config=self.config, experiment_dir="logs")
        self.model.to(self.device)

    def test_initialization(self):
        """Test if the model and its sub-modules are initialized correctly."""
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertIsInstance(self.model.physics_simulator, torch.nn.Module)
        if self.config["ENABLE_ENDORCRINE_SYSTEM"]:
            self.assertIsInstance(self.model.endocrine_system, torch.nn.Module)
        
        # Check embedding dimensions
        self.assertEqual(self.model.position_embeddings.weight.shape, (self.vocab_size, self.config["EMBED_DIM"]))
        self.assertEqual(self.model.velocity_embeddings.weight.shape, (self.vocab_size, 2))
        self.assertEqual(self.model.mass_embeddings.weight.shape, (self.vocab_size, self.config["MASS_EMBEDDING_DIM"]))

    def test_projection_logic(self):
        """Test the internal _project helper method for dimensional transformations."""
        # Test down-projection
        input_512d = torch.randn(10, self.config["EMBED_DIM"], device=self.device)
        output_8d = self.model._project(input_512d, "down_512_to_8")
        self.assertEqual(output_8d.shape, (10, self.config["SEMANTIC_AXIS_COUNT"]))
        
        # Test up-projection
        input_2d = torch.randn(10, 2, device=self.device)
        output_512d = self.model._project(input_2d, "up_2_to_512")
        self.assertEqual(output_512d.shape, (10, self.config["EMBED_DIM"]))

        # Test a multi-step projection
        output_2d = self.model._project(input_512d, "down_512_to_2")
        self.assertEqual(output_2d.shape, (10, 2))

    def test_tidal_level(self):
        """Test the tidal level override mechanism."""
        self.model.set_tidal_level(0.75)
        self.assertEqual(self.model.get_tidal_level().item(), 0.75)
        
        self.model.set_tidal_level(None) # Reset
        self.assertNotEqual(self.model.get_tidal_level().item(), 0.75) # Should revert to clock-based

    def test_forward_pass_output_shapes(self):
        """Test the shapes of tensors returned by the forward pass."""
        batch_size = 4
        seq_len = 20
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        logits, (physics_loss, _), viz_data = self.model(input_ids, target_ids)

        # Check logits shape
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        
        # Check physics loss shape
        self.assertTrue(torch.is_tensor(physics_loss))
        self.assertEqual(physics_loss.numel(), 1)
        
        # Check visualization data dictionary
        self.assertIsInstance(viz_data, dict)
        expected_viz_len = batch_size * seq_len
        self.assertEqual(viz_data['positions_2d'].shape, (expected_viz_len, 2))
        self.assertEqual(viz_data['positions_8d'].shape, (expected_viz_len, self.config["SEMANTIC_AXIS_COUNT"]))
        self.assertEqual(viz_data['masses'].shape, (expected_viz_len, self.config["MASS_EMBEDDING_DIM"]))

    def test_generate_method(self):
        """Test the autoregressive generate method."""
        prompt_ids = torch.tensor([1, 5, 10], device=self.device)
        max_new_tokens = 5
        
        generated_ids = self.model.generate(prompt_ids, max_new_tokens=max_new_tokens)
        
        # Check output type and length
        self.assertIsInstance(generated_ids, list)
        self.assertEqual(len(generated_ids), len(prompt_ids) + max_new_tokens)
        
        # Check that generated token IDs are valid
        for token_id in generated_ids:
            self.assertIsInstance(token_id, int)
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.vocab_size)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
