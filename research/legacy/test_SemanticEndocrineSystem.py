import unittest
import torch
import os

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from SemanticEndocrineSystem import SemanticEndocrineSystem

class TestSemanticEndocrineSystem(unittest.TestCase):
    """
    Unit tests for the SemanticEndocrineSystem class.
    These tests verify the correctness of trigger detection, hormone release/decay,
    and the application of hormonal effects on forces and masses.
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
        """Set up a SemanticEndocrineSystem instance for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure log directory exists for the test logger
        os.makedirs(self.config.get("LOG_DIRECTORY", "logs"), exist_ok=True)
        self.ses = SemanticEndocrineSystem(config=self.config, device=self.device)

    def test_initialization(self):
        """Test if the system is initialized with correct parameters from the config."""
        self.assertEqual(len(self.ses.hormone_types), len(self.config["HORMONE_TYPES"]))
        self.assertEqual(self.ses.hormone_levels.shape[0], len(self.config["HORMONE_TYPES"]))
        self.assertTrue(torch.all(self.ses.hormone_levels == self.config["DEFAULT_HORMONE_INITIAL_LEVEL"]))

    def test_repetition_trigger(self):
        """Test the repetition trigger logic."""
        # High repetition
        token_ids = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], device=self.device)
        embeddings = torch.randn(10, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        triggers = self.ses.detect_triggers(token_ids, embeddings)
        repetition_strength = triggers[self.ses.trigger_map['repetition_trigger']].item()
        self.assertGreater(repetition_strength, self.config["TRIGGER_THRESHOLDS"]["repetition_trigger"])

        # No repetition
        token_ids = torch.arange(10, device=self.device)
        triggers = self.ses.detect_triggers(token_ids, embeddings)
        repetition_strength = triggers[self.ses.trigger_map['repetition_trigger']].item()
        self.assertEqual(repetition_strength, 0.0)

    def test_conceptual_leap_trigger(self):
        """Test the conceptual leap trigger logic."""
        # High leap
        embeddings = torch.tensor([[0.0]*8, [100.0]*8, [0.0]*8, [100.0]*8], dtype=torch.float32, device=self.device)
        token_ids = torch.arange(4, device=self.device)
        triggers = self.ses.detect_triggers(token_ids, embeddings)
        leap_strength = triggers[self.ses.trigger_map['conceptual_leap_trigger']].item()
        self.assertGreater(leap_strength, self.config["TRIGGER_THRESHOLDS"]["conceptual_leap_trigger"])

        # Low leap
        embeddings = torch.randn(10, self.config["SEMANTIC_AXIS_COUNT"], device=self.device) * 0.01
        token_ids = torch.arange(10, device=self.device)
        triggers = self.ses.detect_triggers(token_ids, embeddings)
        leap_strength = triggers[self.ses.trigger_map['conceptual_leap_trigger']].item()
        self.assertLess(leap_strength, self.config["TRIGGER_THRESHOLDS"]["conceptual_leap_trigger"])
    
    def test_creative_mood_trigger(self):
        """Test the creative mood trigger by crafting specific embeddings."""
        v_axis = self.config["SEMANTIC_AXIS_MAPPING"]["V_AXIS"]
        a_axis = self.config["SEMANTIC_AXIS_MAPPING"]["A_AXIS"]
        
        # High Valence and High Arousal should yield a strong creative mood
        embeddings = torch.full((10, self.config["SEMANTIC_AXIS_COUNT"]), -0.5, device=self.device)
        embeddings[:, v_axis] = 0.9 # High Valence
        embeddings[:, a_axis] = 0.9 # High Arousal
        
        token_ids = torch.arange(10, device=self.device)
        triggers = self.ses.detect_triggers(token_ids, embeddings)
        creative_strength = triggers[self.ses.trigger_map['creative_mood_trigger']].item()
        
        self.assertGreater(creative_strength, self.config["TRIGGER_THRESHOLDS"]["creative_mood_trigger"])

    def test_hormone_release(self):
        """Test that a strong trigger releases the correct hormone."""
        catalyst_idx = self.ses.hormone_map['catalyst_hormone']
        self.ses.hormone_levels[catalyst_idx] = 0.0 # Reset level

        # Fabricate triggers to be high for catalyst-releasing moods
        trigger_strengths = torch.zeros(len(self.ses.trigger_names), device=self.device)
        trigger_strengths[self.ses.trigger_map['creative_mood_trigger']] = 1.0

        token_ids = torch.arange(10, device=self.device)
        self.ses.release_hormones(trigger_strengths, torch.tensor(0.0, device=self.device), token_ids)

        self.assertGreater(self.ses.hormone_levels[catalyst_idx].item(), 0.0)

    def test_hormone_decay(self):
        """Test the hormone decay mechanism."""
        self.ses.hormone_levels.fill_(1.0)
        initial_levels = self.ses.hormone_levels.clone()
        
        self.ses.decay_hormones()
        
        self.assertTrue(torch.all(self.ses.hormone_levels < initial_levels))
    
    def test_hormonal_effects_on_forces(self):
        """Test how different hormones modulate 8D forces."""
        forces_8d = torch.ones(5, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        masses = torch.ones(5, self.config["MASS_EMBEDDING_DIM"], device=self.device)
        
        # Test Catalyst hormone effect (creativity boost on affective plane)
        self.ses.hormone_levels.zero_()
        self.ses.hormone_levels[self.ses.hormone_map['catalyst_hormone']] = 1.0
        
        modulated_forces, _ = self.ses.apply_hormonal_effects(forces_8d.clone(), masses)
        
        affective_slice = self.config["SEMANTIC_SPACE_SLICES"]["affective"]
        # Check that affective forces are amplified
        self.assertGreater(torch.mean(modulated_forces[:, affective_slice[0]:affective_slice[1]]).item(), 1.0)
        # Check that other forces are not
        core_slice = self.config["SEMANTIC_SPACE_SLICES"]["core_conceptual"]
        self.assertAlmostEqual(torch.mean(modulated_forces[:, core_slice[0]:core_slice[1]]).item(), 1.0, places=5)

        # Test Stress hormone effect (focus boost on core/interoceptive planes)
        self.ses.hormone_levels.zero_()
        self.ses.hormone_levels[self.ses.hormone_map['stress_hormone']] = 0.5 # Moderate stress
        
        modulated_forces, _ = self.ses.apply_hormonal_effects(forces_8d.clone(), masses)
        
        # Check that core/interoceptive forces are amplified
        self.assertGreater(torch.mean(modulated_forces[:, core_slice[0]:core_slice[1]]).item(), 1.0)
        intero_slice = self.config["SEMANTIC_SPACE_SLICES"]["interoceptive"]
        self.assertGreater(torch.mean(modulated_forces[:, intero_slice[0]:intero_slice[1]]).item(), 1.0)

    def test_hormonal_effects_on_mass(self):
        """Test how different hormones modulate mass if enabled."""
        if not self.config.get("ENABLE_MASS_MODULATION", False):
            self.skipTest("Mass modulation is not enabled in the config.")

        forces_8d = torch.ones(5, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        initial_masses = torch.ones(5, self.config["MASS_EMBEDDING_DIM"], device=self.device)

        # Catalyst should decrease mass
        self.ses.hormone_levels.zero_()
        self.ses.hormone_levels[self.ses.hormone_map['catalyst_hormone']] = 1.0
        _, modulated_masses_catalyst = self.ses.apply_hormonal_effects(forces_8d, initial_masses.clone())
        self.assertLess(torch.mean(modulated_masses_catalyst).item(), 1.0)
        
        # Inhibitor should increase mass
        self.ses.hormone_levels.zero_()
        self.ses.hormone_levels[self.ses.hormone_map['inhibitor_hormone']] = 1.0
        _, modulated_masses_inhibitor = self.ses.apply_hormonal_effects(forces_8d, initial_masses.clone())
        self.assertGreater(torch.mean(modulated_masses_inhibitor).item(), 1.0)

    def test_forward_pass(self):
        """Test a full forward pass of the endocrine system."""
        initial_levels = self.ses.get_hormone_state()
        
        token_ids = torch.randint(0, 100, (20,), device=self.device)
        semantic_positions = torch.randn(20, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        tidal_level = torch.tensor(0.5, device=self.device)
        
        self.ses.forward(token_ids, semantic_positions, tidal_level)
        
        new_levels = self.ses.get_hormone_state()
        
        # Assert that hormone levels have changed after a full cycle.
        self.assertNotEqual(initial_levels, new_levels)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
