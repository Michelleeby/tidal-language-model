import torch
import torch.nn as nn

class Physics2D(nn.Module):
    """
    Handles the 2D physics simulation logic.
    - Uses a direct, vectorized O(N^2) force calculation.
    - Computes a "Grand Unifying" contrastive loss that evaluates relationships
      in the 8D semantic space, driven by the 2D physics.
    """
    def __init__(self, config: dict, vocab_size: int, device: torch.device = None):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Physical constants and parameters
        self.G = nn.Parameter(torch.tensor(self.config["G"], device=self.device))
        self.softening = self.config.get("SOFTENING_PARAM", 1e-6)
        self.epsilon = self.config.get("FORCE_CALCULATION_EPSILON", 1e-9)

        # Learnable centers for semantic regions (wells) in the 8D space.
        self.num_semantic_wells = self.config.get("NUM_SEMANTIC_WELLS", 16)
        self.semantic_well_centers = nn.Parameter(
            torch.randn(self.num_semantic_wells, self.config["SEMANTIC_AXIS_COUNT"], device=self.device)
        )
        self.well_attraction_strength = nn.Parameter(
            torch.tensor(self.config.get("WELL_ATTRACTION_STRENGTH", 0.1), device=self.device)
        )
        self.temperature = nn.Parameter(
            torch.tensor(self.config.get("TEMPERATURE", 1.0), device=self.device)
        )
        self.entropy_coefficient = nn.Parameter(
            torch.tensor(self.config.get("ENTROPY_COEFFICIENT", 0.01), device=self.device)
        )
        self.ke_penalty_factor = self.config.get("KE_PENALTY_FACTOR", 0.001)

        self.repulsion_strength = nn.Parameter(
            torch.tensor(self.config.get("REPULSION_STRENGTH", 0.5), device=self.device)
        )
        self.repulsion_cutoff = self.config.get("REPULSION_CUTOFF", 0.1)

    def calculate_forces(self, positions_2d: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """ 
        Calculates gravitational forces for a BATCH of sequences.
        This batched version prevents memory overflow by computing interactions
        only WITHIN each sequence, not between them.
        Shape of positions_2d: [Batch, Sequence, 2]
        Shape of masses: [Batch, Sequence, 1]
        """
        # B = Batch Size, S = Sequence Length
        # pos_expanded_a shape: [B, S, 1, 2]
        # pos_expanded_b shape: [B, 1, S, 2]
        pos_expanded_a = positions_2d.unsqueeze(2)
        pos_expanded_b = positions_2d.unsqueeze(1)

        # Calculate pairwise differences and squared distances
        # diff shape: [B, S, S, 2]
        # dist_sq shape: [B, S, S]
        diff = pos_expanded_b - pos_expanded_a
        dist_sq = torch.sum(diff.pow(2), dim=-1) + self.softening
        dist = torch.sqrt(dist_sq)
        
        # Create a boolean mask for the diagonal, batched for each sequence
        # diag_mask shape: [S, S]
        diag_mask = torch.eye(positions_2d.shape[1], device=self.device, dtype=torch.bool)

        # Use torch.where to avoid division by zero on the diagonal
        dist_safe = torch.where(diag_mask, torch.ones_like(dist), dist)
        dist_sq_safe = torch.where(diag_mask, torch.ones_like(dist_sq), dist_sq)
       
        # Calculate pairwise force magnitude: F = G * (m1 * m2) / r^2
        # masses_a shape: [B, S, 1, 1]
        # masses_b shape: [B, 1, S, 1]
        masses_a = masses.unsqueeze(2)
        masses_b = masses.unsqueeze(1)
        
        # Squeeze to match dimensions for broadcasting with [B, S, S] dist_sq_safe
        attractive_force_mag = torch.relu(self.G) * (masses_a * masses_b).squeeze(-1) / dist_sq_safe
        
        # Repulsive force calculation
        positive_repulsion_strength = torch.relu(self.repulsion_strength)
        repulsive_force_mag = positive_repulsion_strength * torch.relu(1 - (dist_safe / self.repulsion_cutoff)) / (dist_sq_safe * dist_safe + self.epsilon)

        # Combine forces
        total_force_mag = repulsive_force_mag - attractive_force_mag 
        force_vectors = total_force_mag.unsqueeze(-1) * diff / (dist_safe.unsqueeze(-1) + self.epsilon)
        
        # Sum forces over the second sequence dimension (dim=2) to get the net force on each particle
        total_forces = torch.sum(force_vectors, dim=2)
        
        return total_forces
    
    def verlet_integration(self, positions: torch.Tensor, velocities: torch.Tensor, masses: torch.Tensor, forces: torch.Tensor, tidal_level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Updates 2D positions and velocities using Verlet integration. """
        accelerations = forces / masses
        dt = self.config["DEFAULT_DT"]
        const = self.config["VERLET_INTEGRATION_CONSTANT"]
        new_positions = positions + velocities * dt + const * accelerations * (dt ** 2)
        low_damp = self.config["LOW_TIDE_DAMPING_FACTOR"]
        high_damp = self.config["HIGH_TIDE_DAMPING_FACTOR"]
        damping_factor = low_damp + (high_damp - low_damp) * (tidal_level + 1) / 2
        new_velocities = (velocities + const * accelerations * dt) * damping_factor
        return new_positions, new_velocities

    def calculate_grand_unifying_loss(self,
                                      pos_2d: torch.Tensor,
                                      vel_2d: torch.Tensor,
                                      pos_8d: torch.Tensor,
                                      masses: torch.Tensor,
                                      pos_pos_2d: torch.Tensor,
                                      pos_pos_8d: torch.Tensor,
                                      positive_masses: torch.Tensor,
                                      neg_pos_2d: torch.Tensor,
                                      neg_pos_8d: torch.Tensor,
                                      neg_masses: torch.Tensor,
                                      exp_masses: torch.Tensor
                                     ) -> torch.Tensor:
        """
        Calculates contrastive loss based on Helmholtz Free Energy (F = U - T*S).
        """
        epsilon = self.config["FORCE_CALCULATION_EPSILON"]

        # 1. Calculate Pairwise Potential Energy (U_pair) in 2D space
        dist_pos_2d = torch.norm(pos_2d - pos_pos_2d, dim=-1) + epsilon
        pe_pairwise_pos = -torch.relu(self.G) * (masses * positive_masses).squeeze() / dist_pos_2d

        dist_neg_2d = torch.norm(pos_2d.repeat_interleave(self.config["NUM_NEGATIVE_SAMPLES"], dim=0) - neg_pos_2d, dim=-1) + epsilon
        pe_pairwise_neg = -torch.relu(self.G) * (exp_masses * neg_masses).squeeze() / dist_neg_2d

        # 2. Calculate Well Potential Energy (U_well) in 8D space
        all_pos_8d = torch.cat([pos_8d, pos_pos_8d], dim=0)
        dist_to_wells_pos = torch.cdist(all_pos_8d, self.semantic_well_centers)
        min_dist_pos, _ = torch.min(dist_to_wells_pos, dim=1)

        # Constrain learnable parameters to be non-negative
        positive_well_strength = torch.relu(self.well_attraction_strength)
        positive_temperature = torch.relu(self.temperature)
        positive_entropy_coeff = torch.relu(self.entropy_coefficient)

        pe_well_pos = positive_well_strength * min_dist_pos.pow(2)
        all_neg_8d = torch.cat([pos_8d.repeat_interleave(self.config["NUM_NEGATIVE_SAMPLES"], dim=0), neg_pos_8d], dim=0)
        dist_to_wells_neg = torch.cdist(all_neg_8d, self.semantic_well_centers)
        min_dist_neg, _ = torch.min(dist_to_wells_neg, dim=1)
        pe_well_neg = positive_well_strength * min_dist_neg.pow(2)

        # 3. Combine for Total Internal Energy (U)
        U_pos = pe_pairwise_pos.mean() + pe_well_pos.mean()
        U_neg = pe_pairwise_neg.mean() + pe_well_neg.mean()

        # 4. Calculate Entropy Term (T*S) in 8D space
        batch_variance_8d = torch.mean(torch.var(all_pos_8d, dim=0))
        entropy_term = positive_temperature * positive_entropy_coeff * batch_variance_8d

        # 5. Calculate Helmholtz Free Energy (F = U - T*S)
        F_pos = U_pos - entropy_term
        F_neg = U_neg - entropy_term

        # 6. Calculate Contrastive Loss
        margin = self.config["PHYSICS_LOSS_MARGIN"]
        free_energy_loss = torch.relu(margin + F_pos - F_neg)

        # 7. (Optional) Add Kinetic Energy Penalty in 2D space
        ke_pos = 0.5 * masses * torch.sum(vel_2d.pow(2), dim=-1)
        ke_loss = self.ke_penalty_factor * ke_pos.mean()
        
        # Create a dictionary to hold the loss components
        loss_components = {
            'F_pos': F_pos,
            'F_neg': F_neg
        }

        return free_energy_loss + ke_loss, loss_components