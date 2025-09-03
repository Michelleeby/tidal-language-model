import torch
import torch.nn as nn
import os

from typing import Dict

from Utils import setup_logger

class SemanticEndocrineSystem(nn.Module):
    """
    Simulated endocrine system. Initialized with a configuration
    dictionary, dependent on experiment parameters.
    Maintains system homeostasis by releasing and decaying hormones based on
    sequence and cognitive mood triggers detected in the input.
    """
    
    def __init__(self, config: dict, experiment_dir: str = None, device: torch.device = None):
        super().__init__()
        self.config = config

        log_dir = experiment_dir if experiment_dir else "."
        log_file_training = os.path.join(log_dir, self.config.get("LOG_FILE_TRAINING_ENDORCRINE_SYSTEM", "endocrine-training.log"))
        log_file_evaluation = os.path.join(log_dir, self.config.get("LOG_FILE_EVALUATION_ENDOCRINE_SYSTEM", "endocrine-evaluation.log")) 
        self.logger = setup_logger('TrainingEndocrineSystem', log_file_training, config)
        self.logger_evaluation = setup_logger('EvaluationEndocrineSystem', log_file_evaluation, config)

        self.hormone_types = self.config.get("HORMONE_TYPES", [])
        self.hormone_map = {name: i for i, name in enumerate(self.hormone_types)}

        # Determine device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use register_buffer for state that should be part of the model but not trained.
        initial_levels = torch.full((len(self.hormone_types),), self.config.get("DEFAULT_HORMONE_INITIAL_LEVEL", 0.0), device=self.device)
        self.register_buffer('hormone_levels', initial_levels)

        hormone_initial_states = self.config.get("HORMONE_INITIAL_STATES", {})
        initial_toggles = torch.tensor([hormone_initial_states.get(h, True) for h in self.hormone_types], dtype=torch.bool, device=self.device)
        self.register_buffer('hormone_toggles', initial_toggles)

        self.hormone_baselines = nn.Parameter(torch.full((len(self.hormone_types),), 0.1, device=self.device))
        
        self.trigger_names = list(self.config.get("TRIGGER_THRESHOLDS", {}).keys())
        self.trigger_map = {name: i for i, name in enumerate(self.trigger_names)}
        self.trigger_to_hormone_map = nn.Parameter(torch.tensor([
            # repetition_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['repetition_trigger']],
            # conceptual_leap_trigger -> inhibitor
            [self.hormone_map['inhibitor_hormone'], self.trigger_map['conceptual_leap_trigger']],
            # semantic_diversity_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['semantic_diversity_trigger']],
            # grounding_mood_trigger -> inhibitor
            [self.hormone_map['inhibitor_hormone'], self.trigger_map['grounding_mood_trigger']],
            # creative_mood_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['creative_mood_trigger']],
            # agency_mood_trigger -> stress
            [self.hormone_map['stress_hormone'], self.trigger_map['agency_mood_trigger']],
            # relational_mood_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['relational_mood_trigger']],
            # articulation_mood_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['articulation_mood_trigger']],
            # insight_mood_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['insight_mood_trigger']],
            # synthesis_mood_trigger -> catalyst
            [self.hormone_map['catalyst_hormone'], self.trigger_map['synthesis_mood_trigger']],
        ], dtype=torch.long), requires_grad=False)

        self.activation_sensitivity = self.config.get("ACTIVATION_SENSITIVITY", 2.5)
        self.decay_rate = self.config.get("HORMONE_DECAY_RATE", 0.5)
        
        self.log_counter = 0
        self.log_interval = self.config.get("LOG_INTERVAL_TRAINING_ENDORCRINE_SYSTEM", 1000)
        self.logger.info(f"🧬 Semantic Endocrine System initialized on device: {self.device}.")
    
    # Trigger Detection Methods.

    def detect_triggers(self, token_ids: torch.Tensor, embeddings_8d: torch.Tensor) -> torch.Tensor:
        """Calculates all trigger strengths in a single vectorized operation."""
        # Ensure embeddings are on the correct device, which they should be.
        # Detach to prevent gradients from flowing back through this logic.
        embeddings = embeddings_8d.detach()

        def _normalize(axis_values: torch.Tensor) -> torch.Tensor:
            return (axis_values.mean() + 1) / 2

        # Pre-calculate common values.
        mean_axes = embeddings.mean(dim=0)
        std_axes = torch.std(embeddings, dim=0)

        # Map axis names to their indices from config.
        g, x, v, a, h, s, f, t = [self.config["SEMANTIC_AXIS_MAPPING"][f"{axis}_AXIS"] for axis in "GXVHASFT"]

        # Define all of our triggers.

        # Cognitive Mood Triggers.
        grounding_strength = 1.0 - _normalize(embeddings[:, g])
        homeostasis_strength = _normalize(embeddings[:, h])
        grounding_mood = (grounding_strength + homeostasis_strength) / 2
        valence_strength = _normalize(embeddings[:, v])
        arousal_strength = _normalize(embeddings[:, a])
        creative_mood = (valence_strength + arousal_strength) / 2
        agency_mood = (arousal_strength + _normalize(embeddings[:, s])) / 2
        relational_mood = (valence_strength + (1.0 - arousal_strength)) / 2
        articulation_mood = 1.0 - torch.abs(mean_axes[f])
        abstraction_strength = _normalize(embeddings[:, g])
        specificity_variance = torch.std(embeddings[:, x])
        insight_mood = (abstraction_strength + specificity_variance) / 2
        synthesis_mood = (abstraction_strength + (1.0 - _normalize(embeddings[:, x]))) / 2

        # Repetition Trigger.
        if len(token_ids) > self.config["REPETITION_TRIGGER_TOKEN_SEQUENCE_THRESHOLD"]:
            repetition_ratio = 1.0 - (torch.unique(token_ids).size(0) / len(token_ids))
            threshold = self.trigger_configs['repetition_trigger']
            repetition_trigger = torch.relu((repetition_ratio - threshold) / (1.0 - threshold + 1e-6))
        else:
            repetition_trigger = torch.tensor(0.0, device=self.device)

        # Conceptual Leap Trigger.
        if embeddings.shape[0] < 2:
            conceptual_leap = torch.tensor(0.0, device=self.device)
        else:
            distances = torch.norm(embeddings[1:] - embeddings[:-1], p=2, dim=1)
            conceptual_leap = torch.mean(distances)

        # Semantic Diversity Trigger.
        if embeddings.shape[0] < 2:
            semantic_diversity = torch.tensor(0.0, device=self.device)
        else:
            semantic_diversity = torch.mean(std_axes)

        # Stack all defined triggers into a single tensor in the correct order.
        # NOTE: The order here MUST match the order in `self.trigger_names` from the `__init__` method.
        trigger_strengths = torch.stack([
            repetition_trigger,
            conceptual_leap,
            semantic_diversity,
            grounding_mood,
            creative_mood,
            agency_mood,
            relational_mood,
            articulation_mood,
            insight_mood,
            synthesis_mood
        ])

        return trigger_strengths

    # Endocrine Cycle Methods.

    def release_hormones(self, trigger_strengths: torch.Tensor, current_tidal_level: torch.Tensor):
        """
        Vectorized method for releasing hormones based on trigger strengths.
        """
        if trigger_strengths is None: return

        # 1. Gather the strengths and thresholds for each hormone-trigger pair
        hormone_indices = self.trigger_to_hormone_map[:, 0]
        trigger_indices = self.trigger_to_hormone_map[:, 1]

        strengths = trigger_strengths[trigger_indices]

        # Gather thresholds using the trigger names list
        threshold_values = torch.stack([self.trigger_configs[name] for name in self.trigger_names])
        thresholds = threshold_values[trigger_indices]

        # 2. Apply tidal bias
        tidal_bias = torch.ones_like(strengths)
        # High tide boosts catalyst hormone
        catalyst_mask = (hormone_indices == self.hormone_map['catalyst_hormone']) & (current_tidal_level > 0)
        tidal_bias[catalyst_mask] *= (1.0 + current_tidal_level * self.config["HIGH_TIDE_CATALYST_HORMONE_RELEASE_AMOUNT"])
        # Low tide boosts inhibitor hormone
        inhibitor_mask = (hormone_indices == self.hormone_map['inhibitor_hormone']) & (current_tidal_level < 0)
        tidal_bias[inhibitor_mask] *= (1.0 - current_tidal_level * self.config["LOW_TIDE_INHIBITOR_HORMONE_RELEASE_AMOUNT"])

        strengths *= tidal_bias

        # 3. Calculate activation for all triggers that are above their threshold
        # torch.relu effectively replaces the 'if strength < threshold' check
        activations = torch.sigmoid(self.activation_sensitivity * torch.relu(strengths - thresholds))
        release_amounts = activations * self.config["HORMONE_MAX_RELEASE_AMOUNT"]

        # 4. Apply suppression effect
        inhibitor_level = self.hormone_levels[self.hormone_map['inhibitor_hormone']]
        if inhibitor_level > self.config["INHIBITOR_HORMONE_SUPPRESSION_THRESHOLD"]:
            suppression_factor = 1.0 - inhibitor_level
            release_amounts[catalyst_mask] *= suppression_factor # Apply suppression only to catalyst releases

        # 5. Aggregate releases for each hormone
        # Create a zero tensor for total releases per hormone
        total_release = torch.zeros_like(self.hormone_levels)
        # Add the calculated releases to the correct hormone's slot.
        # This is a vectorized way of doing: for h, r in zip(hormones, releases): total[h] += r
        total_release.scatter_add_(0, hormone_indices, release_amounts)

        # 6. Update hormone levels and clamp
        self.hormone_levels += total_release
        torch.clamp_(self.hormone_levels, 0.0, 1.0)

    def apply_hormonal_effects(self, forces_8d: torch.Tensor, masses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized method for applying hormonal effects to forces and masses in the semantic space.
        """
        modulated_forces_8d = forces_8d.clone()
        catalyst_level = self.hormone_levels[self.hormone_map['catalyst_hormone']]
        inhibitor_level = self.hormone_levels[self.hormone_map['inhibitor_hormone']]
        stress_level = self.hormone_levels[self.hormone_map['stress_hormone']]
        
        # Antagonistic hormone effects. Catalyst (estrogen-like) is dampened by inhibitor (progesterone-like).
        net_creativity_effect = 1.0 + torch.relu((catalyst_level * self.config["CATALYST_HORMONE_EFFECT"]) - (inhibitor_level * self.config["INHIBITOR_HORMONE_EFFECT"]))
        # This calculates a multiplier that is 1.0 if the effect is <= 1.0, and the effect value otherwise.
        creativity_multiplier = torch.relu(net_creativity_effect - 1.0) + 1.0
        affective_slice = self.config["SEMANTIC_SPACE_SLICES"]["affective"]
        modulated_forces_8d[:, affective_slice[0]:affective_slice[1]] *= creativity_multiplier

        # Non-linear stress hormone effects.

        # Moderate stress (cortisol-like) enhances focus on core logic.
        stress_boost = 1.0 + stress_level * self.config["STRESS_HORMONE_EFFECT"]
        core_slice = self.config["SEMANTIC_SPACE_SLICES"]["core_conceptual"]
        interoceptive_slice = self.config["SEMANTIC_SPACE_SLICES"]["interoceptive"]
        
        # We only apply the boost if stress is above a minimum level. This 'if' is acceptable as it's a single
        # check per batch, but for consistency we can use relu.
        is_stress_active = torch.relu(stress_level - self.config["HORMONE_MIN_ACTIVE_LEVEL"]) > 0
        if is_stress_active:
            modulated_forces_8d[:, core_slice[0]:core_slice[1]] *= stress_boost
            modulated_forces_8d[:, interoceptive_slice[0]:interoceptive_slice[1]] *= stress_boost

        # Inverted-U: Excessive stress impairs creativity.
        excess_stress = torch.relu(stress_level - self.config["STRESS_HORMONE_EXCESS_THRESHOLD"])
        impairment_factor = (excess_stress / (1.0 - self.config["STRESS_HORMONE_EXCESS_THRESHOLD"] + 1e-6)) * self.config["STRESS_HORMONE_IMPAIRMENT_FACTOR"]
        impairment_multiplier = 1.0 - impairment_factor
        affective_slice = self.config["SEMANTIC_SPACE_SLICES"]["affective"]
        structural_slice = self.config["SEMANTIC_SPACE_SLICES"]["structural"]
        modulated_forces_8d[:, affective_slice[0]:affective_slice[1]] *= impairment_multiplier
        modulated_forces_8d[:, structural_slice[0]:structural_slice[1]] *= impairment_multiplier

        # Hormone levels also modulate the mass of a concept in the semantic space.
        # This gives them dynamic, mood-driven "weight".
        modulated_masses = masses.clone()
        if self.config.get("ENABLE_MASS_MODULATION", False):
            catalyst_multiplier = 1.0 + (catalyst_level * self.config.get("CATALYST_HORMONE_MASS_EFFECT", 0.0))
            inhibitor_multiplier = 1.0 + (inhibitor_level * self.config.get("INHIBITOR_HORMONE_MASS_EFFECT", 0.0))
            stress_multiplier = 1.0 + (stress_level * self.config.get("STRESS_HORMONE_MASS_EFFECT", 0.0))
            final_multiplier = catalyst_multiplier * inhibitor_multiplier * stress_multiplier
            modulated_masses = masses * final_multiplier
            modulated_masses = torch.clamp(modulated_masses, min=self.config.get("MASS_MODULATION_MIN_MASS", 1e-4))

        max_force = self.config["MAX_FORCE"]
        if self.config.get("STRESS_HORMONE_AFFECTS_MAX_FORCE", False) and stress_level > 0:
            max_force *= (1 + stress_level)

        clamped_forces_8d = torch.clamp(modulated_forces_8d, -max_force, max_force)
        return clamped_forces_8d, modulated_masses

    def decay_hormones(self):
        with torch.no_grad():
            baselines = torch.clamp(self.hormone_baselines, 0, 1)
            self.hormone_levels.mul_(self.decay_rate).add_(baselines * (1 - self.decay_rate))

    def get_hormone_state(self) -> Dict[str, float]:
        return {h: self.hormone_levels[self.hormone_map[h]].item() for h in self.hormone_types}
    
    def forward(self, token_ids: torch.Tensor, semantic_positions: torch.Tensor, current_tidal_level: torch.Tensor) -> None:
        trigger_strengths = self.detect_triggers(token_ids, semantic_positions)
        self.release_hormones(trigger_strengths, current_tidal_level)
        self.decay_hormones()