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
    def __init__(self, vocab_size: int, config: dict, experiment_dir: str = None, log_buffer: list = None):
        super().__init__()
        self.config = config
        self.experiment_dir = experiment_dir
        self.vocab_size = vocab_size
        self.log_buffer = log_buffer
        self.embed_dim = self.config.get("EMBED_DIM", 512)

        device_str = self.config.get("DEVICE", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        
        self.enable_resonance = self.config["ENABLE_RESONANCE"]
        self.enable_endocrine_system = self.config["ENABLE_ENDORCRINE_SYSTEM"]
        
        self.physics_simulator = Physics2D(self.config, self.vocab_size, device=self.device)
        self.G = self.physics_simulator.G # Share the learnable parameter
        
        self.position_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.velocity_embeddings = nn.Embedding(vocab_size, 2) # For 2D physical space
        self.mass_embeddings = nn.Embedding(vocab_size, self.config["MASS_EMBEDDING_DIM"])
        self._initialize_embeddings()

        self.projection_512d_to_8d = nn.Linear(self.embed_dim, self.config["SEMANTIC_AXIS_COUNT"])
        self.projection_8d_to_2d = nn.Linear(self.config["SEMANTIC_AXIS_COUNT"], 2)
        self.projection_8d_to_512d = nn.Linear(self.config["SEMANTIC_AXIS_COUNT"], self.embed_dim)
        self.projection_2d_to_8d = nn.Linear(2, self.config["SEMANTIC_AXIS_COUNT"])

        self.projection_plans = {
            "down_512_to_2": [self.projection_512d_to_8d, self.projection_8d_to_2d],
            "up_2_to_512":   [self.projection_2d_to_8d, self.projection_8d_to_512d],
            "up_2_to_8":     [self.projection_2d_to_8d],
            "down_512_to_8": [self.projection_512d_to_8d],
            "down_8_to_2":   [self.projection_8d_to_2d],
            "up_8_to_512":   [self.projection_8d_to_512d]
        }

        self.semantic_gru_hidden_size = self.config["SEMANTIC_AXIS_COUNT"]
        self.gru = nn.GRU(self.config["SEMANTIC_AXIS_COUNT"], self.semantic_gru_hidden_size, batch_first=True)
        self.output_projection = nn.Linear(self.embed_dim, vocab_size)

        if self.enable_endocrine_system:
            self.endocrine_system = SemanticEndocrineSystem(
                config=self.config, 
                experiment_dir=self.experiment_dir, 
                device=self.device,
                log_buffer=self.log_buffer 
            )

        if self.enable_resonance:
            self.resonance_strength = nn.Parameter(torch.tensor(self.config["RESONANCE_STRENGTH"]))
            self.pattern_embeddings = nn.Embedding(vocab_size, self.config["SEMANTIC_AXIS_COUNT"])

        self.register_buffer('global_step', torch.tensor(0, dtype=torch.float))
        self.circadian_period = self.config["CIRCADIAN_PERIOD"]
        self.tidal_level_override = None

    def get_tidal_level(self):
        """Calculates the current tidal level based on the internal clock."""
        if self.tidal_level_override is not None:
            return torch.tensor(self.tidal_level_override, device=self.device)
        return torch.sin(2 * math.pi * self.global_step / self.circadian_period)

    def set_tidal_level(self, level: float = None):
        """For evaluation: overrides the internal clock to force a specific tidal level."""
        self.tidal_level_override = level
    
    def _project(self, input_tensor: torch.Tensor, plan_name: str) -> torch.Tensor:
        """Projects a tensor through a sequence of layers defined in a plan."""
        output_tensor = input_tensor
        plan = self.projection_plans.get(plan_name)
        if plan is None:
            raise KeyError(f"Projection plan '{plan_name}' not found.")
        for layer in plan:
            output_tensor = layer(output_tensor)
        return output_tensor

    def _project_delta(self, original_tensor: torch.Tensor, old_state: torch.Tensor, new_state: torch.Tensor, plan_name: str) -> torch.Tensor:
        """Calculates a delta, projects it, and adds it to an original tensor."""
        delta_state = new_state - old_state
        projected_delta = self._project(delta_state, plan_name)
        return original_tensor + projected_delta

    def _initialize_embeddings(self):
        nn.init.normal_(self.position_embeddings.weight, mean=self.config["POSITION_EMBEDDING_MEAN"], std=self.config["POSITION_EMBEDDING_STD"])
        nn.init.zeros_(self.velocity_embeddings.weight)
        nn.init.uniform_(self.mass_embeddings.weight, self.config["MASS_EMBEDDING_MIN"], self.config["MASS_EMBEDDING_MAX"])

    def _apply_local_resonance(self, positions: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Applies semantic resonance in 8D to reinforce repeated concepts, modulated by tides."""
        if not self.enable_resonance: return positions

        positions_with_resonance = positions.clone()
        tidal_level = self.get_tidal_level()
        low_res = self.config["LOW_TIDE_RESONANCE_MASK"]
        high_res = self.config["HIGH_TIDE_RESONANCE_MASK"]
        resonance_mask = low_res + (high_res - low_res) * (tidal_level + 1) / 2
        
        unique_tokens, counts = torch.unique(token_ids, return_counts=True)
        repeated_tokens = unique_tokens[counts > 1]
        
        for token_id in repeated_tokens:
            pattern_emb = self.pattern_embeddings(token_id)
            mask = (token_ids == token_id)
            positions_with_resonance[mask] += resonance_mask * self.resonance_strength * pattern_emb * 0.1
            
        return positions_with_resonance

    def _update_endocrine_system(self, token_ids: torch.Tensor):
        """Updates the endocrine system based on the current semantic state."""
        if self.enable_endocrine_system:
            with torch.no_grad():
                tidal_level = self.get_tidal_level()
                semantic_positions = self._project(self.position_embeddings(token_ids), plan_name="down_512_to_8")
            self.endocrine_system(token_ids, semantic_positions, tidal_level)

    def _get_initial_state(self, token_ids: torch.Tensor):
        """Retrieves the initial positions, velocities, and masses from embedding tables."""
        positions = self.position_embeddings(token_ids)
        velocities_2d = self.velocity_embeddings(token_ids)
        masses = self.mass_embeddings(token_ids)
        return positions, velocities_2d, masses

    def _apply_biological_effects(self, raw_forces_2d: torch.Tensor, masses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies hormonal effects from the Semantic Endocrine System to modulate forces and masses.
        This acts as a bridge between the raw physics and the biological simulation layer.
        """
        if self.enable_endocrine_system:
            raw_forces_8d = self._project(raw_forces_2d, plan_name="up_2_to_8")
            final_forces_8d, effective_masses = self.endocrine_system.apply_hormonal_effects(raw_forces_8d, masses)
            final_forces_2d = self._project(final_forces_8d, plan_name="down_8_to_2")
        else:
            final_forces_2d = torch.clamp(raw_forces_2d, -self.config["MAX_FORCE"], self.config["MAX_FORCE"])
            effective_masses = masses
        return final_forces_2d, effective_masses

    def _calculate_physics_loss(self, token_ids, context_word_ids, sim_data):
        """Calculates the Grand Unifying Loss if conditions are met."""
        if not (self.training and context_word_ids is not None and self.config["ENABLE_PHYSICS_LOSS"]):
            return torch.tensor(0.0, device=self.device), None

        positions_2d, velocities_2d, positions_8d, masses = sim_data
        pos_pos_embed = self.position_embeddings(context_word_ids)
        pos_pos_8d = self._project(pos_pos_embed, plan_name="down_512_to_8")
        pos_pos_2d = self._project(pos_pos_8d, plan_name="down_8_to_2")
        positive_masses = self.mass_embeddings(context_word_ids)
        num_neg = self.config["NUM_NEGATIVE_SAMPLES"]
        neg_ids = torch.randint(0, self.vocab_size, (len(token_ids) * num_neg,), device=self.device) 
        neg_pos_embed = self.position_embeddings(neg_ids)
        neg_pos_8d = self._project(neg_pos_embed, plan_name="down_512_to_8")
        neg_pos_2d = self._project(neg_pos_8d, plan_name="down_8_to_2")
        neg_masses = self.mass_embeddings(neg_ids)

        loss, loss_components = self.physics_simulator.calculate_grand_unifying_loss(
            positions_2d, velocities_2d, positions_8d, masses,
            pos_pos_2d, pos_pos_8d, positive_masses,
            neg_pos_2d, neg_pos_8d, neg_masses,
            masses.repeat_interleave(num_neg, dim=0)
        )

        return loss, loss_components

    def _update_global_step(self):
        """Updates the global step of the simulation."""
        if self.training and self.tidal_level_override is None:
            self.global_step = self.global_step + 1

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 40, tidal_level: float = None, repetition_penalty: float = 1.2):
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
        if tidal_level is not None: self.set_tidal_level(tidal_level)
        
        # Start with the prompt and keep the tensor on the GPU.
        generated_tokens = prompt_ids.to(self.device).view(-1)
        
        with torch.no_grad():
            # Initial state from prompt
            positions_512d, velocities_2d, masses = self._get_initial_state(generated_tokens)
            positions_8d = self._project(positions_512d, "down_512_to_8")
            hidden_state = None

            # "Prime" the GRU with the full prompt sequence
            if len(generated_tokens) > 0:
                b_positions_8d = positions_8d.unsqueeze(0)
                b_velocities_2d = velocities_2d.unsqueeze(0)
                b_masses = masses.unsqueeze(0)

                b_positions_2d = self._project(b_positions_8d, plan_name="down_8_to_2")
                b_raw_forces_2d = self.physics_simulator.calculate_forces(b_positions_2d, b_masses)
                b_final_forces_2d, b_effective_masses = self._apply_biological_effects(b_raw_forces_2d, b_masses)
                b_new_pos_2d, b_new_velocities_2d = self.physics_simulator.verlet_integration(
                    b_positions_2d, b_velocities_2d, b_effective_masses, b_final_forces_2d, self.get_tidal_level()
                )
                
                new_pos_2d = b_new_pos_2d.squeeze(0)
                pos_2d = b_positions_2d.squeeze(0)
                
                prompt_embeddings_8d = self._project_delta(positions_8d, pos_2d, new_pos_2d, plan_name="up_2_to_8")
                _, hidden_state = self.gru(prompt_embeddings_8d.unsqueeze(0), hidden_state)

            # Autoregressive generation loop
            for _ in range(max_new_tokens):
                self._update_endocrine_system(generated_tokens)

                # Get the state of the last token to predict the next one
                last_pos_8d = positions_8d[-1:].unsqueeze(0)
                last_vel_2d = velocities_2d[-1:].unsqueeze(0)
                last_mass = masses[-1:].unsqueeze(0)

                # Simulate physics for the single last token
                b_positions_2d = self._project(last_pos_8d, plan_name="down_8_to_2")
                # Force calculation needs context, so we use the full sequence here.
                full_pos_2d = self._project(positions_8d.unsqueeze(0), plan_name="down_8_to_2")
                full_masses = masses.unsqueeze(0)
                b_raw_forces_2d = self.physics_simulator.calculate_forces(full_pos_2d, full_masses)[:, -1:]
                
                b_final_forces_2d, b_effective_masses = self._apply_biological_effects(b_raw_forces_2d, last_mass)
                
                b_new_pos_2d, b_new_velocities_2d = self.physics_simulator.verlet_integration(
                    b_positions_2d, last_vel_2d, b_effective_masses, b_final_forces_2d, self.get_tidal_level()
                )

                # Project delta and update GRU state
                new_pos_8d_last = self._project_delta(last_pos_8d.squeeze(0), b_positions_2d.squeeze(0), b_new_pos_2d.squeeze(0), plan_name="up_2_to_8")
                gru_output_8d, hidden_state = self.gru(new_pos_8d_last.unsqueeze(0), hidden_state)
                
                # Get logits for the next token
                gru_output_512d = self._project(gru_output_8d, plan_name="up_8_to_512")
                logits = self.output_projection(gru_output_512d.squeeze(1))

                # Apply repetition penalty
                if len(generated_tokens) > 0:
                    unique_tokens = torch.unique(generated_tokens)
                    logits[0, unique_tokens] /= repetition_penalty
                
                # Sampling
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_k_val = min(top_k, self.vocab_size)
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k_val)
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, 1, next_token_idx).squeeze().view(1)

                # Append the new token and its state to our running tensors on the GPU
                generated_tokens = torch.cat((generated_tokens, next_token_id))
                
                new_pos_512d = self.position_embeddings(next_token_id)
                new_pos_8d_next = self._project(new_pos_512d, "down_512_to_8")
                new_vel_2d_next = self.velocity_embeddings(next_token_id)
                new_mass_next = self.mass_embeddings(next_token_id)
                
                # Update the full state tensors
                positions_8d = torch.cat((positions_8d, new_pos_8d_next), dim=0)
                velocities_2d = torch.cat((velocities_2d, new_vel_2d_next), dim=0)
                masses = torch.cat((masses, new_mass_next), dim=0)
                
                if tidal_level is None: self._update_global_step()

        self.set_tidal_level(None)
        self.train() 
        # Convert to a standard Python list on the CPU only once at the end.
        return generated_tokens.cpu().tolist()

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
        """
        Orchestrates the full forward pass of the model.
        
        To efficiently simulate our embedding space,
        first we project from 512D to our 8D semantic space. Then,
        this is projected to a 2D physical space where the 
        physics simulation is applied. Then, we project back
        to the 8D semantic space. Here the endocrine, tide,
        and sequential memory (GRU) systems affect the space.
        This final 8D state is projected to 512D for output.
         
        We can imagine this like:
        
        -> [input] -- 
        -> (Embedding Space 512D) ---> -- [projection_512d_to_8d] -- 
        -> (Semantic Space 8D) ---> -- [projection_8d_to_2d] -- 
        -> (Physical Space 2D) ---> -- {apply physics simulation} ---> -- [projection_2d_to_8d] -- 
        -> (Semantic Space 8D) ---> -- {apply Endocrine + Tide + GRU} ---> -- [projection_8d_to_512d] --
        -> (Output Space 512D) ---> -- [output] --
        """
        batch_size, seq_len = input_ids.shape

        self._update_endocrine_system(input_ids)
        positions_512d, velocities_2d, masses = self._get_initial_state(input_ids)
        positions_8d = self._project(positions_512d, plan_name="down_512_to_8")
        positions_2d = self._project(positions_8d, plan_name="down_8_to_2")
        raw_forces_2d = self.physics_simulator.calculate_forces(positions_2d, masses)
        final_forces_2d, effective_masses = self._apply_biological_effects(raw_forces_2d, masses)
        
        new_pos_2d, new_velocities_2d = self.physics_simulator.verlet_integration(
            positions_2d, velocities_2d, effective_masses, final_forces_2d, self.get_tidal_level()
        )
        new_positions_8d = self._project_delta(positions_8d, positions_2d, new_pos_2d, plan_name="up_2_to_8")
        final_positions_8d_flat = self._apply_local_resonance(new_positions_8d.view(-1, self.semantic_gru_hidden_size), input_ids.view(-1))
        
        final_embeddings_8d = final_positions_8d_flat.view(batch_size, seq_len, self.semantic_gru_hidden_size)
        gru_output_8d, _ = self.gru(final_embeddings_8d)
        output_512d = self._project(gru_output_8d, plan_name="up_8_to_512")
        logits = self.output_projection(output_512d)

        sim_data_for_loss = (positions_2d.view(-1, 2), 
                             new_velocities_2d.view(-1, 2), 
                             positions_8d.view(-1, self.config["SEMANTIC_AXIS_COUNT"]), 
                             effective_masses.view(-1, self.config["MASS_EMBEDDING_DIM"]))
        physics_loss, physics_loss_components = self._calculate_physics_loss(input_ids.view(-1), target_ids.view(-1) if target_ids is not None else None, sim_data_for_loss)
        
        self._update_global_step()

        viz_data = {
            'positions_2d': new_pos_2d.view(-1, 2).detach(),
            'positions_8d': new_positions_8d.view(-1, self.config["SEMANTIC_AXIS_COUNT"]).detach(),
            'forces_2d': final_forces_2d.view(-1, 2).detach(),
            'masses': effective_masses.view(-1, self.config["MASS_EMBEDDING_DIM"]).detach()
        }

        return logits, (physics_loss, physics_loss_components), viz_data