import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional, List, Any

class ConstantLanguageModel(nn.Module):
    """
    Standard transformer-based language model for baseline comparison with Tidal Language Model.

    This model serves as a control in research experiments to validate the effectiveness
    of physics-based and bio-inspired features. It uses standard multi-headed self-attention
    and feed-forward networks without any physics simulation, semantic spaces, or endocrine systems.

    Architecture:
        Input Tokens → Token Embeddings (512D) + Position Embeddings →
        → Transformer Blocks (Multi-Head Attention + FFN) × N →
        → Layer Normalization → Output Projection → Next Token Prediction

    The model is designed to be config-driven and maintain interface compatibility
    with TidalLanguageModel for seamless integration with existing training infrastructure.
    """

    def __init__(self, vocab_size: int, config: dict, experiment_dir: str = None, log_buffer: list = None):
        """
        Initialize the Constant Language Model.

        Args:
            vocab_size: Size of the vocabulary
            config: Configuration dictionary with model hyperparameters
            experiment_dir: Directory for experiment artifacts (for interface compatibility)
            log_buffer: Shared logging buffer (for interface compatibility)
        """
        super().__init__()
        self.config = config
        self.experiment_dir = experiment_dir
        self.vocab_size = vocab_size
        self.log_buffer = log_buffer

        # Model dimensions from config
        self.embed_dim = self.config.get("EMBED_DIM", 512)
        self.num_transformer_blocks = self.config.get("NUM_TRANSFORMER_BLOCKS", 6)
        self.num_attention_heads = self.config.get("NUM_ATTENTION_HEADS", 8)
        self.ffn_hidden_dim = self.config.get("FFN_HIDDEN_DIM", 2048)
        self.dropout = self.config.get("DROPOUT", 0.1)
        self.max_context_length = self.config.get("MAX_CONTEXT_LENGTH", 512)

        # Device setup
        device_str = self.config.get("DEVICE", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.position_embeddings = nn.Embedding(self.max_context_length, self.embed_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_attention_heads,
                ffn_hidden_dim=self.ffn_hidden_dim,
                dropout=self.dropout,
                max_seq_len=self.max_context_length
            )
            for _ in range(self.num_transformer_blocks)
        ])

        # Final layer normalization and output projection
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, vocab_size)

        # Initialize weights
        self._initialize_weights()

        # Move model to device
        self.to(self.device)

    def _initialize_weights(self):
        """Initialize model weights using standard techniques."""
        # Token embeddings: normal initialization
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)

        # Position embeddings: normal initialization
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

        # Output projection: normal initialization with smaller std
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[Dict]], Dict]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            target_ids: Target token IDs for loss calculation (batch_size, seq_len)
            return_hidden_states: If True, include hidden states in viz_data

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss_tuple: (total_loss, loss_components_dict) - loss_components_dict is None for constant model
            viz_data: Dict with optional hidden_states for RL observation
        """
        batch_size, seq_len = input_ids.shape

        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)

        # Combine embeddings and apply dropout
        hidden_states = self.embedding_dropout(token_embeds + pos_embeds)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        # Final layer norm and output projection
        final_hidden = self.final_layer_norm(hidden_states)
        logits = self.output_projection(final_hidden)

        # Calculate loss if targets provided (for training)
        total_loss = torch.tensor(0.0, device=self.device)
        if target_ids is not None and self.training:
            # Standard cross-entropy loss
            # Reshape for loss calculation: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = target_ids.view(-1)

            total_loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=-100  # Standard padding ignore index
            )

        # Return in format compatible with TidalLanguageModel
        # No physics loss components for constant model
        loss_tuple = (total_loss, None)

        # Include hidden states for RL observation if requested
        viz_data = {}
        if return_hidden_states:
            viz_data["hidden_states"] = final_hidden

        return logits, loss_tuple, viz_data

    def forward_with_hidden(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both logits and hidden states.
        Convenience method for RL observation extraction.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Final layer hidden states of shape (batch_size, seq_len, embed_dim)
        """
        logits, _, viz_data = self.forward(input_ids, return_hidden_states=True)
        return logits, viz_data["hidden_states"]

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = 40, tidal_level: float = None, repetition_penalty: float = 1.2):
        """
        Generate text autoregressively from a prompt.

        Args:
            prompt_ids: Initial prompt token IDs (1D tensor)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            tidal_level: Ignored (for interface compatibility with TidalLanguageModel)
            repetition_penalty: Penalty factor for repeated tokens

        Returns:
            List of generated token IDs
        """
        self.eval()

        # Start with the prompt
        generated_tokens = prompt_ids.to(self.device).view(-1)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get the context window (last max_context_length tokens)
                context = generated_tokens[-self.max_context_length:]

                # Forward pass through model
                input_ids = context.unsqueeze(0)  # Add batch dimension
                logits, _, _ = self.forward(input_ids)

                # Get logits for the last position (next token prediction)
                next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

                # Apply repetition penalty
                if len(generated_tokens) > 0:
                    unique_tokens = torch.unique(generated_tokens)
                    next_token_logits[unique_tokens] /= repetition_penalty

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Convert to probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                # Top-k sampling
                top_k_val = min(top_k, self.vocab_size)
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k_val)

                # Sample from top-k distribution
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[next_token_idx]

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token_id])

        self.train()
        return generated_tokens.cpu().tolist()

    def generate_with_hormones(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        hormone_policy: Any,
        modulator: Any,
        base_temperature: float = 1.0,
        top_k: int = 40,
        return_trajectory: bool = False
    ) -> Tuple[List[int], Optional[Dict]]:
        """
        Generate text with RL-controlled hormone modulation.

        This method generates text while using an RL policy to control hormone
        levels at each step. The hormones modulate temperature, repetition penalty,
        and attention focus.

        Args:
            prompt_ids: Initial prompt token IDs (1D tensor)
            max_new_tokens: Maximum number of tokens to generate
            hormone_policy: RL policy with get_action(observation) method
            modulator: HormoneModulator instance
            base_temperature: Base temperature before hormone modulation
            top_k: Keep only top k tokens for sampling
            return_trajectory: If True, return trajectory data for training

        Returns:
            generated_tokens: List of generated token IDs
            trajectory: Dict with observations, actions, rewards (if return_trajectory=True)
        """
        self.eval()

        # Start with the prompt
        generated_tokens = prompt_ids.to(self.device).view(-1)

        # Trajectory storage for RL training
        trajectory = {
            "observations": [],
            "actions": [],
            "logits_history": [],
            "tokens": [],
            "hidden_states": []
        } if return_trajectory else None

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get the context window (last max_context_length tokens)
                context = generated_tokens[-self.max_context_length:]

                # Forward pass with hidden states
                input_ids = context.unsqueeze(0)  # Add batch dimension
                logits, hidden_states = self.forward_with_hidden(input_ids)

                # Build observation for RL policy
                observation = self._build_rl_observation(
                    generated_tokens,
                    hidden_states,
                    logits,
                    step,
                    max_new_tokens
                )

                # Get hormone action from policy
                if hasattr(hormone_policy, 'get_action'):
                    hormones = hormone_policy.get_action(observation)
                else:
                    # Assume it's a callable
                    hormones = hormone_policy(observation)

                # Apply hormone effects
                effects = modulator(hormones, len(context), self.device)

                # Get logits for the last position (next token prediction)
                next_token_logits = logits[0, -1, :].clone()  # Shape: (vocab_size,)

                # Apply repetition penalty via modulator
                next_token_logits = modulator.apply_repetition_penalty_to_logits(
                    next_token_logits,
                    generated_tokens,
                    effects.repetition_penalty
                )

                # Apply temperature
                next_token_logits = next_token_logits / effects.temperature

                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Top-k sampling
                top_k_val = min(top_k, self.vocab_size)
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k_val)

                # Sample from top-k distribution
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[next_token_idx]

                # Store trajectory data if requested
                if trajectory is not None:
                    trajectory["observations"].append(observation.clone())
                    trajectory["actions"].append(hormones.clone() if isinstance(hormones, torch.Tensor) else torch.tensor(hormones))
                    trajectory["logits_history"].append(logits[0, -1, :].clone())
                    trajectory["tokens"].append(next_token_id.item())
                    trajectory["hidden_states"].append(hidden_states[0, -1, :].clone())

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token_id])

        self.train()
        return generated_tokens.cpu().tolist(), trajectory

    def _build_rl_observation(
        self,
        generated_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        step: int,
        max_steps: int
    ) -> torch.Tensor:
        """
        Build observation vector for RL policy.

        Observation space (64D):
        - Token statistics (10D): repetition ratio, entropy, n-gram diversity
        - Hidden state summary (48D): mean/std pooling from final layer, attention entropy
        - Context (6D): temperature, step position, previous hormone levels (placeholder)

        Args:
            generated_tokens: All generated tokens so far
            hidden_states: Hidden states from forward pass (batch, seq, embed)
            logits: Output logits (batch, seq, vocab)
            step: Current generation step
            max_steps: Maximum generation steps

        Returns:
            Observation tensor of shape (64,)
        """
        device = hidden_states.device
        obs_parts = []

        # --- Token Statistics (10D) ---
        tokens = generated_tokens.cpu().tolist()
        num_tokens = len(tokens)

        # 1. Repetition ratio (fraction of repeated tokens)
        if num_tokens > 0:
            unique_ratio = len(set(tokens)) / num_tokens
            repetition_ratio = 1.0 - unique_ratio
        else:
            repetition_ratio = 0.0

        # 2. Token entropy (from recent distribution)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        token_entropy = -(probs * (probs + 1e-10).log()).sum().item()
        token_entropy = token_entropy / 10.0  # Normalize roughly

        # 3. Bigram diversity (unique bigrams / total bigrams)
        if num_tokens > 1:
            bigrams = [(tokens[i], tokens[i+1]) for i in range(num_tokens - 1)]
            bigram_diversity = len(set(bigrams)) / len(bigrams)
        else:
            bigram_diversity = 1.0

        # 4. Trigram diversity
        if num_tokens > 2:
            trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(num_tokens - 2)]
            trigram_diversity = len(set(trigrams)) / len(trigrams)
        else:
            trigram_diversity = 1.0

        # 5-10. Recent token statistics (last 10 tokens)
        recent_tokens = tokens[-10:] if len(tokens) >= 10 else tokens
        if len(recent_tokens) > 0:
            recent_unique_ratio = len(set(recent_tokens)) / len(recent_tokens)
            # Frequency of most common recent token
            from collections import Counter
            counter = Counter(recent_tokens)
            most_common_freq = counter.most_common(1)[0][1] / len(recent_tokens)
        else:
            recent_unique_ratio = 1.0
            most_common_freq = 0.0

        token_stats = torch.tensor([
            repetition_ratio,
            token_entropy,
            bigram_diversity,
            trigram_diversity,
            recent_unique_ratio,
            most_common_freq,
            min(num_tokens / 100.0, 1.0),  # Normalized sequence length
            0.0, 0.0, 0.0  # Padding for 10D
        ], device=device)
        obs_parts.append(token_stats)

        # --- Hidden State Summary (48D) ---
        # Use last position hidden state
        last_hidden = hidden_states[0, -1, :]  # (embed_dim,)

        # Mean pooling over embedding dimension (chunked for 24D)
        chunk_size = self.embed_dim // 24
        hidden_mean = torch.zeros(24, device=device)
        for i in range(24):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            hidden_mean[i] = last_hidden[start_idx:end_idx].mean()

        # Std pooling over embedding dimension (chunked for 24D)
        hidden_std = torch.zeros(24, device=device)
        for i in range(24):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            hidden_std[i] = last_hidden[start_idx:end_idx].std()

        obs_parts.append(hidden_mean)
        obs_parts.append(hidden_std)

        # --- Context Features (6D) ---
        step_progress = step / max(max_steps, 1)  # Progress through generation
        context_features = torch.tensor([
            step_progress,
            1.0,  # Base temperature (placeholder)
            0.5, 0.5, 0.5,  # Previous hormones (placeholder, filled by env)
            0.0  # Padding
        ], device=device)
        obs_parts.append(context_features)

        # Concatenate all parts
        observation = torch.cat(obs_parts)

        # Ensure exactly 64D
        if observation.shape[0] < 64:
            padding = torch.zeros(64 - observation.shape[0], device=device)
            observation = torch.cat([observation, padding])
        elif observation.shape[0] > 64:
            observation = observation[:64]

        return observation


class TransformerBlock(nn.Module):
    """
    Standard transformer block with multi-headed self-attention and feed-forward network.
    Includes residual connections and layer normalization.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int, dropout: float = 0.1, max_seq_len: int = 512):
        """
        Initialize a transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ffn_hidden_dim: Hidden dimension in feed-forward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for causal mask caching
        """
        super().__init__()

        # Multi-headed self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),  # GELU activation (common in modern transformers)
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization (pre-norm architecture)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Pre-compute and cache causal mask for max sequence length
        # Register as buffer so it moves with the model to GPU
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Pre-layer norm architecture
        # Multi-headed self-attention with residual connection
        normalized = self.ln1(x)

        # Use cached causal mask, sliced to current sequence length
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Apply attention
        attn_output, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_output)

        # Feed-forward network with residual connection
        normalized = self.ln2(x)
        ffn_output = self.ffn(normalized)
        x = x + self.dropout2(ffn_output)

        return x
