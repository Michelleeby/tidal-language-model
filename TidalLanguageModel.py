import torch
import torch.nn as nn
import math

from SemanticEndocrineSystem import SemanticEndocrineSystem
from Physics2D import Physics2D

class TidalLanguageModel(nn.Module):
    """
    # Tidal Language Model

    A physics-based, bio-inspired language model that learns to understand the 
    relationships between higher-dimensional concepts in a lower-dimensional 
    semantic space.

    ## Semantic Space

    The Tidal Language Model's "Semantic Space" can be conceptualized like:

    ### I. Core Conceptual Plane

    This plane defines the intrinsic nature of a concept.

    1. G-Axis (Groundedness): Measures the concept's tie to direct 
    sensory-motor experience.  
        * Low G: Concrete & Sensory (rock, water, heavy)
        * High G: Abstract & Relational (justice, theory, freedom)  
    2. X-Axis (Taxonomic Specificity): Measures the concept's 
    level in a hierarchy.  
        * Low X: General & Superordinate (animal, tool, idea)  
        * High X: Specific & Subordinate (poodle, screwdriver, sonnet)

    ### II. Affective Plane (Emotion)

    This plane maps the core emotional qualities 
    based on the Circumplex Model.

    3. V-Axis (Valence): Measures the positive or negative emotional charge.  
        * Low V: Negative (pain, sad, anger)  
        * High V: Positive (joy, peace, beauty)  
    4. A-Axis (Arousal): Measures the emotional intensity or energy.  
        * Low A: Deactivated & Calm (sleep, bored, serene)  
        * High A: Activated & Excited (panic, rage, surprise)

    ### III. Interoceptive Plane (Bodily State)

    This plane maps the underlying physiological states, inspired by 
    the gut-brain axis.

    5. H-Axis (Homeostasis): Measures the sense of internal 
    balance versus dysregulation.
        * Low H: Dysregulated (nausea, fatigue, stress)  
        * High H: Regulated (healthy, rested, energized)  
    6. S-Axis (Somatic Focus): Measures if a concept is experienced more 
    in the body or mind.  
        * Low S: Cognitive (belief, memory, curiosity)  
        * High S: Somatic/Physical (pain, itch, warmth)

    ### IV. Structural Plane (Contextual Role)

    This plane defines the concept's role within a larger structure like a sentence
    or timeline.

    7. F-Axis (Functional Role): Measures a word's purpose as either 
    providing meaning or grammatical structure.  
        * Low F: Lexical/Content (mountain, sun, ocean)  
        * High F: Functional/Grammar (is, the, because)  
    8. T-Axis (Temporal Orientation): Measures a concept's inherent 
    relationship to time.  
        * Low T: Past-Oriented (history, memory, yesterday)  
        * High T: Future-Oriented (plan, hope, tomorrow)
    """
    def __init__(self, vocab_size: int, config: dict, experiment_dir: str = None):
        super().__init__()
        self.config = config
        self.experiment_dir = experiment_dir
        self.vocab_size = vocab_size
        self.embed_dim = self.config["EMBED_DIM"]
        
        # Determine device
        device_str = self.config.get("DEVICE", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.enable_resonance = self.config["ENABLE_RESONANCE"]
        self.enable_endocrine_system = self.config["ENABLE_ENDORCRINE_SYSTEM"]
        
        # Physics Simulator
        self.physics_simulator = Physics2D(self.config, self.vocab_size, device=self.device)
        self.G = self.physics_simulator.G # Share the learnable parameter
        
        # Model components.
        self.position_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.velocity_embeddings = nn.Embedding(vocab_size, 2) # For 2D physical space
        self.mass_embeddings = nn.Embedding(vocab_size, self.config["MASS_EMBEDDING_DIM"])
        self._initialize_embeddings()
        
        # Projection Layers.
        # Project from real-world space to 8D semantic space.
        self.projection_layer = nn.Linear(self.embed_dim, self.config["SEMANTIC_AXIS_COUNT"])
        # Project from 8D semantic space to 2D physical space.
        self.projection_to_2d = nn.Linear(self.config["SEMANTIC_AXIS_COUNT"], 2)
        # Project from 8D semantic space to real-world space.
        self.inverse_projection_layer = nn.Linear(self.config["SEMANTIC_AXIS_COUNT"], self.embed_dim)
        # Project forces from 2D physical space back to 8D semantic space.
        self.inverse_projection_from_2d = nn.Linear(2, self.config["SEMANTIC_AXIS_COUNT"])
        # Project to real-world space.
        self.output_projection = nn.Linear(self.embed_dim, vocab_size)

        # Endocrine System
        if self.enable_endocrine_system:
            self.endocrine_system = SemanticEndocrineSystem(config=self.config, experiment_dir=self.experiment_dir, device=self.device)

        # Resonance Components
        if self.enable_resonance:
            self.resonance_strength = nn.Parameter(torch.tensor(self.config["RESONANCE_STRENGTH"]))
            self.pattern_embeddings = nn.Embedding(vocab_size, self.embed_dim)

        # Internal state for tides
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.float))
        self.circadian_period = self.config["CIRCADIAN_PERIOD"]
        self.tidal_level_override = None

    def _initialize_embeddings(self):
        nn.init.normal_(self.position_embeddings.weight, mean=self.config["POSITION_EMBEDDING_MEAN"], std=self.config["POSITION_EMBEDDING_STD"])
        nn.init.zeros_(self.velocity_embeddings.weight)
        nn.init.uniform_(self.mass_embeddings.weight, self.config["MASS_EMBEDDING_MIN"], self.config["MASS_EMBEDDING_MAX"])

    def _apply_local_resonance(self, positions: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Applies semantic resonance to reinforce repeated concepts, modulated by tides."""
        if not self.enable_resonance: return positions

        tidal_level = self.get_tidal_level()
        low_res = self.config["LOW_TIDE_RESONANCE_MASK"]
        high_res = self.config["HIGH_TIDE_RESONANCE_MASK"]
        resonance_mask = low_res + (high_res - low_res) * (tidal_level + 1) / 2
        
        unique_tokens, counts = torch.unique(token_ids, return_counts=True)
        repeated_tokens = unique_tokens[counts > 1]
        
        for token_id in repeated_tokens:
            pattern_emb = self.pattern_embeddings(token_id)
            mask = (token_ids == token_id)
            positions[mask] += resonance_mask * self.resonance_strength * pattern_emb * 0.1
        return positions

    def get_tidal_level(self):
        """Calculates the current tidal level based on the internal clock."""
        if self.tidal_level_override is not None:
            return torch.tensor(self.tidal_level_override, device=self.device)
        return torch.sin(2 * math.pi * self.global_step / self.circadian_period)

    def set_tidal_level(self, level: float = None):
        """For evaluation: overrides the internal clock to force a specific tidal level."""
        self.tidal_level_override = level

    def _update_endocrine_system(self, token_ids: torch.Tensor):
        """Updates the endocrine system based on the current semantic state."""
        if self.enable_endocrine_system:
            with torch.no_grad():
                tidal_level = self.get_tidal_level()
                semantic_positions = self.projection_layer(self.position_embeddings(token_ids))
            self.endocrine_system(token_ids, semantic_positions, tidal_level)

    def _get_initial_state(self, token_ids: torch.Tensor):
        """Retrieves the initial positions, velocities, and masses from embedding tables."""
        positions = self.position_embeddings(token_ids)
        velocities_2d = self.velocity_embeddings(token_ids)
        masses = self.mass_embeddings(token_ids)
        return positions, velocities_2d, masses

    def _run_physics_simulation(self, positions_512d: torch.Tensor, velocities_2d: torch.Tensor, masses: torch.Tensor):
        """Orchestrates the core physics simulation across different vector spaces."""
        # Project positions down to 2D for efficient simulation
        positions_8d = self.projection_layer(positions_512d)
        positions_2d = self.projection_to_2d(positions_8d)

        # Run physics simulation.
        raw_forces_2d = self.physics_simulator.calculate_forces(positions_2d, masses)

        # Apply hormonal effects and clamping
        if self.enable_endocrine_system:
            raw_forces_8d = self.inverse_projection_from_2d(raw_forces_2d)
            final_forces_8d, effective_masses = self.endocrine_system.apply_hormonal_effects(raw_forces_8d, masses)
            final_forces_2d = self.projection_to_2d(final_forces_8d)
        else:
            final_forces_2d = torch.clamp(raw_forces_2d, -self.config["MAX_FORCE"], self.config["MAX_FORCE"])
            effective_masses = masses

        # Integrate physics in 2D space
        new_positions_2d, new_velocities_2d = self.physics_simulator.verlet_integration(
            positions_2d, velocities_2d, effective_masses, final_forces_2d, self.get_tidal_level()
        )
        return new_positions_2d, new_velocities_2d, positions_2d, positions_8d, final_forces_2d, effective_masses

    def _calculate_new_positions(self, positions_512d: torch.Tensor, positions_2d: torch.Tensor, new_positions_2d: torch.Tensor):
        """Projects the change in position from 2D back to the original embedding space."""
        position_delta_2d = new_positions_2d - positions_2d
        position_delta_8d = self.inverse_projection_from_2d(position_delta_2d)
        position_delta_embed = self.inverse_projection_layer(position_delta_8d)
        return positions_512d + position_delta_embed

    def _calculate_physics_loss(self, token_ids, context_word_ids, sim_data):
        """Calculates the Grand Unifying Loss if conditions are met."""
        if not (self.training and context_word_ids is not None and self.config["ENABLE_PHYSICS_LOSS"]):
            # Always return a tuple to maintain a consistent API
            return torch.tensor(0.0, device=self.device), None

        # Unpack simulation data
        positions_2d, velocities_2d, positions_8d, masses = sim_data

        # Get positive context words and project them
        pos_pos_embed = self.position_embeddings(context_word_ids)
        pos_pos_8d = self.projection_layer(pos_pos_embed)
        pos_pos_2d = self.projection_to_2d(pos_pos_8d)
        positive_masses = self.mass_embeddings(context_word_ids)

        # Negative sampling
        num_neg = self.config["NUM_NEGATIVE_SAMPLES"]
        neg_ids = torch.randint(0, self.vocab_size, (len(token_ids) * num_neg,), device=self.device)
        
        # Get negative samples and project them
        neg_pos_embed = self.position_embeddings(neg_ids)
        neg_pos_8d = self.projection_layer(neg_pos_embed)
        neg_pos_2d = self.projection_to_2d(neg_pos_8d)
        neg_masses = self.mass_embeddings(neg_ids)

        # Calculate the loss from the simulator
        loss = self.physics_simulator.calculate_grand_unifying_loss(
            positions_2d, velocities_2d, positions_8d, masses,
            pos_pos_2d, pos_pos_8d, positive_masses,
            neg_pos_2d, neg_pos_8d, neg_masses,
            masses.repeat_interleave(num_neg, dim=0)
        )
    
        # Always return a tuple to maintain a consistent API
        return loss, None
    
    def _update_global_step(self):
        """Updates the global step of the simulation."""
        if self.training and self.tidal_level_override is None:
            self.global_step += 1

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 40, tidal_level: float = None):
        """
        Generates a sequence of tokens auto-regressively, maintaining the physics state.
        Args:
            prompt_ids (torch.Tensor): A tensor of token IDs for the initial prompt (e.g., shape [N]).
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Softmax temperature for sampling. Higher values -> more randomness.
            top_k (int): Filters predictions to the top K most likely tokens before sampling.
            tidal_level (float, optional): Manually set the tidal level for generation mood. 
                                            If None, uses the internal clock. Defaults to None.
        """
        self.eval()
        if tidal_level is not None:
            self.set_tidal_level(tidal_level)

        prompt_ids = prompt_ids.to(self.device).view(-1)
        generated_ids = list(prompt_ids.cpu().numpy())
        
        with torch.no_grad():
            positions, velocities_2d, masses = self._get_initial_state(prompt_ids)

            for _ in range(max_new_tokens):
                current_sequence_ids = torch.tensor(generated_ids, dtype=torch.long, device=self.device)
                self._update_endocrine_system(current_sequence_ids)

                new_pos_2d, new_velocities_2d, pos_2d, _, _, effective_masses = self._run_physics_simulation(
                    positions, velocities_2d, masses
                )
                new_positions = self._calculate_new_positions(positions, pos_2d, new_pos_2d)

                last_token_position = new_positions[-1:]
                logits = self.output_projection(last_token_position)

                # Prevent semantic space collapse. (When concepts get too close, they repel).
                if len(generated_ids) > 0:
                    # Create a tensor of unique generated ids on the correct device.
                    recent_tokens = torch.tensor(list(set(generated_ids)), device=self.device, dtype=torch.long)
                    # Use scatter_ to apply the penalty by division (in log space, this is subtraction).
                    # We divide the logits for penalized tokens, making them less likely.
                    logits.scatter_(1, recent_tokens.unsqueeze(1), logits.gather(1, recent_tokens.unsqueeze(1)) / repetition_penalty)
                
                logits = logits / temperature
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                top_k_val = min(top_k, self.vocab_size)
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k_val)
                
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, 1, next_token_idx).squeeze()

                generated_ids.append(next_token_id.item())
                
                new_position_embedding = self.position_embeddings(next_token_id.unsqueeze(0))
                new_velocity_embedding = self.velocity_embeddings(next_token_id.unsqueeze(0))
                new_mass_embedding = self.mass_embeddings(next_token_id.unsqueeze(0))
                
                positions = torch.cat((new_positions, new_position_embedding), dim=0)
                velocities_2d = torch.cat((new_velocities_2d, new_velocity_embedding), dim=0)
                masses = torch.cat((effective_masses, new_mass_embedding), dim=0)

                # Optional: Add velocity decay to prevent runaway energy.
                velocities_2d *= 0.98 
                
                if tidal_level is None:
                    self._update_global_step()

        self.set_tidal_level(None)
        self.train() 
        return generated_ids

    def forward(self, token_ids: torch.Tensor, context_word_ids: torch.Tensor = None):
        """Orchestrates the full forward pass of the model."""
        # 1. Update bio-inspired systems.
        self._update_endocrine_system(token_ids)

        # 2. Run core physics simulation.
        positions, velocities_2d, masses = self._get_initial_state(token_ids)
        new_pos_2d, new_velocities_2d, pos_2d, pos_8d, final_forces_2d, effective_masses = self._run_physics_simulation(
            positions, velocities_2d, masses
        )
        new_positions = self._calculate_new_positions(positions, pos_2d, new_pos_2d)

        # 3. Predict next token.
        final_batch_positions = self._apply_local_resonance(new_positions, token_ids)
        logits = self.output_projection(final_batch_positions)

        # 4. Calculate physics-based training loss.
        sim_data_for_loss = (pos_2d, new_velocities_2d, pos_8d, effective_masses)
        physics_loss, _ = self._calculate_physics_loss(token_ids, context_word_ids, sim_data_for_loss)

        # 5. Update global clock.
        self._update_global_step()

        # 6. Prepare visualization data.
        # Using .detach() is important to avoid holding onto the computation graph.
        viz_data = {
            'positions_2d': new_pos_2d.detach(),
            'positions_8d': pos_8d.detach(),
            'forces_2d': final_forces_2d.detach(),
            'masses': effective_masses.detach()
        }

        return logits, physics_loss, viz_data