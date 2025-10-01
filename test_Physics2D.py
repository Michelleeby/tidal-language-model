import unittest
import torch
import os

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from Physics2D import Physics2D

class TestPhysics2D(unittest.TestCase):
    """
    Unit tests for the Physics2D class.
    These tests verify the correctness of force calculations, integration,
    and the loss function using the project's unified configuration file.
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
        """Set up a Physics2D instance for each test using the loaded config."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.physics = Physics2D(config=self.config, vocab_size=100, device=self.device)

    def test_initialization(self):
        """Test if the Physics2D module is initialized with correct parameters."""
        self.assertEqual(self.physics.G.item(), self.config["G"])
        self.assertEqual(self.physics.softening, self.config["SOFTENING_PARAM"])
        self.assertEqual(self.physics.semantic_well_centers.shape, (self.config["NUM_SEMANTIC_WELLS"], self.config["SEMANTIC_AXIS_COUNT"]))
        self.assertTrue(self.physics.G.requires_grad)
        self.assertTrue(self.physics.semantic_well_centers.requires_grad)

    def test_calculate_forces_two_body_attraction(self):
        """Test the attractive force between two particles."""
        # Batch of 1, sequence of 2 particles
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], device=self.device)
        masses = torch.tensor([[[1.0], [1.0]]], device=self.device)

        forces = self.physics.calculate_forces(positions, masses)
        
        # Particle 1 (at origin) should be pulled towards particle 2 (positive x direction)
        self.assertGreater(forces[0, 0, 0].item(), 0)
        self.assertAlmostEqual(forces[0, 0, 1].item(), 0.0, places=5)
        
        # Particle 2 should be pulled towards particle 1 (negative x direction)
        self.assertLess(forces[0, 1, 0].item(), 0)
        self.assertAlmostEqual(forces[0, 1, 1].item(), 0.0, places=5)

        # Newton's third law: F_12 = -F_21
        self.assertTrue(torch.allclose(forces[0, 0], -forces[0, 1]))
        
        # Check magnitude: F = G*m1*m2 / r^2
        expected_force_mag = self.config["G"] * (1.0 * 1.0) / (1.0**2)
        # Repulsion is zero at this distance, so total force mag should equal attractive force mag
        self.assertAlmostEqual(torch.norm(forces[0, 0]).item(), expected_force_mag, places=5)

    def test_calculate_forces_repulsion(self):
        """Test the repulsive force when two particles are very close."""
        positions = torch.tensor([[[0.0, 0.0], [0.05, 0.0]]], device=self.device) # Closer than REPULSION_CUTOFF from config
        masses = torch.tensor([[[1.0], [1.0]]], device=self.device)

        forces = self.physics.calculate_forces(positions, masses)
        
        # Particle 1 (at origin) should be pushed away from particle 2 (negative x direction)
        self.assertLess(forces[0, 0, 0].item(), 0)
        # Particle 2 should be pushed away from particle 1 (positive x direction)
        self.assertGreater(forces[0, 1, 0].item(), 0)

    def test_calculate_forces_no_self_force(self):
        """Ensure a particle exerts no force on itself."""
        positions = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], device=self.device)
        masses = torch.tensor([[[1.0], [1.0]]], device=self.device)
        
        # Manually calculate force of particle 0 on 1 and vice-versa
        diff = positions[0, 1] - positions[0, 0]
        dist_sq = torch.sum(diff.pow(2))
        dist = torch.sqrt(dist_sq)
        # Attraction should be dominant here as they are far apart
        force_mag = self.config["G"] * (1.0 * 1.0) / dist_sq
        force_vec = force_mag * diff / dist
        
        forces = self.physics.calculate_forces(positions, masses)
        
        # The calculated force on particle 1 should be due to particle 0 only.
        self.assertTrue(torch.allclose(forces[0, 1], -force_vec)) # Force on 1 from 0 is opposite to force on 0 from 1

    def test_verlet_integration(self):
        """Test the Verlet integration for updating positions and velocities."""
        dt = self.config["DEFAULT_DT"]
        const = self.config["VERLET_INTEGRATION_CONSTANT"]

        # Case 1: No force
        positions = torch.zeros((1, 1, 2), device=self.device)
        velocities = torch.ones((1, 1, 2), device=self.device)
        masses = torch.ones((1, 1, 1), device=self.device)
        forces = torch.zeros((1, 1, 2), device=self.device)
        tidal_level = torch.tensor(0.0, device=self.device)

        new_pos, new_vel = self.physics.verlet_integration(positions, velocities, masses, forces, tidal_level)
        
        expected_pos = positions + velocities * dt
        self.assertTrue(torch.allclose(new_pos, expected_pos))
        
        # Case 2: Constant force (constant acceleration)
        forces = torch.tensor([[[1.0, 0.0]]], device=self.device)
        accelerations = forces / masses
        
        expected_new_pos = positions + velocities * dt + const * accelerations * (dt ** 2)
        
        damping = self.config["LOW_TIDE_DAMPING_FACTOR"] + (self.config["HIGH_TIDE_DAMPING_FACTOR"] - self.config["LOW_TIDE_DAMPING_FACTOR"]) * (tidal_level + 1) / 2
        expected_new_vel = (velocities + const * accelerations * dt) * damping
        
        new_pos, new_vel = self.physics.verlet_integration(positions, velocities, masses, forces, tidal_level)
        
        self.assertTrue(torch.allclose(new_pos, expected_new_pos))
        self.assertTrue(torch.allclose(new_vel, expected_new_vel))

    def test_tidal_damping_in_verlet(self):
        """Test that damping is applied according to tidal level as defined in config."""
        positions = torch.zeros((1, 1, 2), device=self.device)
        velocities = torch.ones((1, 1, 2), device=self.device)
        masses = torch.ones((1, 1, 1), device=self.device)
        forces = torch.ones((1, 1, 2), device=self.device)
        
        # High tide has a higher damping factor (less damping) in the config
        high_tide = torch.tensor(1.0, device=self.device)
        _, new_vel_high = self.physics.verlet_integration(positions, velocities, masses, forces, high_tide)
        
        # Low tide has a lower damping factor (more damping) in the config
        low_tide = torch.tensor(-1.0, device=self.device)
        _, new_vel_low = self.physics.verlet_integration(positions, velocities, masses, forces, low_tide)

        # Velocity magnitude should be greater at high tide due to less damping
        self.assertGreater(torch.norm(new_vel_high), torch.norm(new_vel_low))

    def test_grand_unifying_loss(self):
        """Test the structure and output of the grand unifying loss function."""
        N = 10 # Number of particles
        
        # Dummy data for the loss function
        pos_2d = torch.randn(N, 2, device=self.device)
        vel_2d = torch.randn(N, 2, device=self.device)
        pos_8d = torch.randn(N, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        masses = torch.rand(N, 1, device=self.device) + 0.1
        
        pos_pos_2d = torch.randn(N, 2, device=self.device)
        pos_pos_8d = torch.randn(N, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        positive_masses = torch.rand(N, 1, device=self.device) + 0.1
        
        num_neg = self.config["NUM_NEGATIVE_SAMPLES"]
        neg_pos_2d = torch.randn(N * num_neg, 2, device=self.device)
        neg_pos_8d = torch.randn(N * num_neg, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        neg_masses = torch.rand(N * num_neg, 1, device=self.device) + 0.1
        exp_masses = masses.repeat_interleave(num_neg, dim=0)

        loss, loss_components = self.physics.calculate_grand_unifying_loss(
            pos_2d, vel_2d, pos_8d, masses,
            pos_pos_2d, pos_pos_8d, positive_masses,
            neg_pos_2d, neg_pos_8d, neg_masses, exp_masses
        )

        # Loss should be a single scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.numel(), 1)
        
        # Loss components should be a dictionary with expected keys
        self.assertIsInstance(loss_components, dict)
        self.assertIn('F_pos', loss_components)
        self.assertIn('F_neg', loss_components)

if __name__ == '__main__':
    # You might need to install PyYAML: pip install pyyaml
    unittest.main(argv=['first-arg-is-ignored'], exit=False)