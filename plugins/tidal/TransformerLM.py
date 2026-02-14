import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from typing import Tuple, Dict, Optional, List, Any


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for Rotary Positional Embeddings.

    Args:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency for the rotation angles.

    Returns:
        (cos, sin) each of shape (max_seq_len, head_dim).
    """
    # Frequencies for pairs: theta^(-2i/d) for i in [0, head_dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Position indices
    positions = torch.arange(max_seq_len).float()
    # Outer product: (max_seq_len, head_dim/2)
    angles = torch.outer(positions, freqs)
    # Duplicate each angle so shape becomes (max_seq_len, head_dim)
    angles = angles.repeat(1, 2)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2, ...] → [-x_{d/2+1}, ..., x1, ...]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding to a tensor.

    Args:
        x: (batch, heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        Tensor of same shape with positional information encoded.
    """
    # Broadcast cos/sin to (1, 1, seq_len, head_dim)
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (_rotate_half(x) * sin.unsqueeze(0).unsqueeze(0))


class DynamicGate(nn.Module):
    """
    Small MLP that converts 3D gate signals → per-dimension scaling factors.

    Maps [creativity, focus, stability] → sigmoid output of shape (embed_dim,).
    Initialized so that sigmoid output ≈ 1.0 at start (neutral / no effect).
    """

    def __init__(self, gate_dim: int, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Bias the final linear so sigmoid(output) ≈ 1.0 at init
        with torch.no_grad():
            self.net[-1].bias.fill_(2.0)
            self.net[-1].weight.zero_()

    def forward(self, gate_signals: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gate_signals: (batch, 3) or None.
        Returns:
            scale: (batch, 1, embed_dim) — broadcastable over seq_len.
                   All-ones when gate_signals is None.
        """
        if gate_signals is None:
            return 1.0
        scale = torch.sigmoid(self.net(gate_signals))  # (batch, embed_dim)
        return scale.unsqueeze(1)  # (batch, 1, embed_dim)


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-norm attention + FFN and residual connections.
    Uses manual Q/K/V projections with RoPE and optional KV cache.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        gate_signals: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch, seq_len, embed_dim = x.shape
        normalized = self.ln1(x)

        q = self.q_proj(normalized).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normalized).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normalized).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        kv_len = k.size(2)
        causal_mask = torch.triu(torch.ones(seq_len, kv_len, device=x.device), diagonal=kv_len - seq_len + 1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        x = x + self.dropout1(attn_output)

        normalized = self.ln2(x)
        ffn_output = self.ffn(normalized)
        x = x + self.dropout2(ffn_output)

        if use_cache:
            return x, present
        return x


class GatedTransformerBlock(nn.Module):
    """
    Transformer block with DynamicGate modules that scale attention and FFN outputs
    based on 3D gate signals [creativity, focus, stability].

    Uses manual Q/K/V projections with RoPE and optional KV cache.
    When gate_signals is None, gates produce unit scaling (no effect).
    """

    GATE_DIM = 3

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Dynamic gates
        self.attn_gate = DynamicGate(self.GATE_DIM, embed_dim)
        self.ffn_gate = DynamicGate(self.GATE_DIM, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        gate_signals: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch, seq_len, embed_dim = x.shape
        normalized = self.ln1(x)

        q = self.q_proj(normalized).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normalized).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normalized).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        kv_len = k.size(2)
        causal_mask = torch.triu(torch.ones(seq_len, kv_len, device=x.device), diagonal=kv_len - seq_len + 1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        if gate_signals is not None:
            attn_output = attn_output * self.attn_gate(gate_signals)
        x = x + self.dropout1(attn_output)

        normalized = self.ln2(x)
        ffn_output = self.ffn(normalized)
        if gate_signals is not None:
            ffn_output = ffn_output * self.ffn_gate(gate_signals)
        x = x + self.dropout2(ffn_output)

        if use_cache:
            return x, present
        return x


class TransformerLM(nn.Module):
    """
    Gated Transformer language model (~20M parameters).

    Architecture:
        Input tokens → Token Embeddings (256D) + Position Embeddings
        → 6 × GatedTransformerBlock(gate_signals)
        → LayerNorm → Output Projection → Logits

    Gate signals (3D): [creativity, focus, stability]
        Provided by a PPO agent (GatingPolicyAgent).
        Each block has DynamicGate modules that convert 3 signals → per-dim scaling.
        When gate_signals=None, gates produce unit scaling (no effect).
    """

    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Model dimensions from config
        self.embed_dim = config.get("EMBED_DIM", 256)
        self.num_transformer_blocks = config.get("NUM_TRANSFORMER_BLOCKS", 6)
        self.num_attention_heads = config.get("NUM_ATTENTION_HEADS", 8)
        self.ffn_hidden_dim = config.get("FFN_HIDDEN_DIM", 1024)
        self.dropout = config.get("DROPOUT", 0.1)
        self.max_context_length = config.get("MAX_CONTEXT_LENGTH", 512)

        # Device setup
        device_str = config.get("DEVICE", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # Embeddings (RoPE replaces learned position embeddings)
        self.token_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)

        # Precompute RoPE frequency buffers
        self.head_dim = self.embed_dim // self.num_attention_heads
        rope_cos, rope_sin = precompute_rope_frequencies(self.head_dim, self.max_context_length)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Gated transformer blocks
        self.transformer_blocks = nn.ModuleList([
            GatedTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_attention_heads,
                ffn_hidden_dim=self.ffn_hidden_dim,
                dropout=self.dropout,
                max_seq_len=self.max_context_length,
            )
            for _ in range(self.num_transformer_blocks)
        ])

        # Final layer normalization and output projection
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, vocab_size)

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
        gate_signals: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[Dict]], Dict]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            target_ids: Target token IDs for loss calculation (batch_size, seq_len)
            gate_signals: Optional (batch_size, 3) tensor of [creativity, focus, stability]
            return_hidden_states: If True, include hidden states in viz_data
            use_cache: If True, return KV cache in viz_data for incremental decoding
            past_key_values: Cached KV pairs from previous forward pass

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss_tuple: (total_loss, None)
            viz_data: Dict with optional hidden_states and past_key_values
        """
        batch_size, seq_len = input_ids.shape

        # Compute position offset from past KV cache
        if past_key_values is not None:
            position_offset = past_key_values[0][0].shape[2]  # past K seq_len
        else:
            position_offset = 0

        token_embeds = self.token_embeddings(input_ids)
        hidden_states = self.embedding_dropout(token_embeds)

        # Slice RoPE tables for current positions
        rope_cos = self.rope_cos[position_offset:position_offset + seq_len]
        rope_sin = self.rope_sin[position_offset:position_offset + seq_len]

        presents = [] if use_cache else None
        for i, block in enumerate(self.transformer_blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = block(
                    hidden_states, gate_signals,
                    rope_cos=rope_cos, rope_sin=rope_sin,
                    layer_past=layer_past, use_cache=True,
                )
                presents.append(present)
            else:
                hidden_states = block(
                    hidden_states, gate_signals,
                    rope_cos=rope_cos, rope_sin=rope_sin,
                    layer_past=layer_past,
                )

        final_hidden = self.final_layer_norm(hidden_states)
        logits = self.output_projection(final_hidden)

        # Calculate loss if targets provided (both train and eval)
        total_loss = torch.tensor(0.0, device=input_ids.device)
        if target_ids is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = target_ids.view(-1)
            total_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        loss_tuple = (total_loss, None)

        viz_data = {}
        if return_hidden_states:
            viz_data["hidden_states"] = final_hidden
        if use_cache:
            viz_data["past_key_values"] = presents

        return logits, loss_tuple, viz_data

    def forward_with_hidden(
        self,
        input_ids: torch.Tensor,
        gate_signals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits and hidden states.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            gate_signals: Optional (batch_size, 3) gate signals

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden_states: (batch_size, seq_len, embed_dim)
        """
        logits, _, viz_data = self.forward(input_ids, gate_signals=gate_signals, return_hidden_states=True)
        return logits, viz_data["hidden_states"]

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ):
        """
        Generate text autoregressively from a prompt using KV cache.

        Args:
            prompt_ids: Initial prompt token IDs (1D tensor)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            repetition_penalty: Penalty factor for repeated tokens

        Returns:
            List of generated token IDs
        """
        self.eval()
        generated_tokens = prompt_ids.to(self.device).view(-1)

        with torch.no_grad():
            # Prefill: process entire prompt, get initial KV cache
            context = generated_tokens[-self.max_context_length:]
            input_ids = context.unsqueeze(0)
            logits, _, viz_data = self.forward(input_ids, use_cache=True)
            past_key_values = viz_data["past_key_values"]

            for _ in range(max_new_tokens):
                next_token_logits = logits[0, -1, :].clone()

                if len(generated_tokens) > 0:
                    unique_tokens = torch.unique(generated_tokens)
                    next_token_logits[unique_tokens] /= repetition_penalty

                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)

                top_k_val = min(top_k, self.vocab_size)
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k_val)
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[next_token_idx]

                generated_tokens = torch.cat([generated_tokens, next_token_id])

                # Decode: single token with cached KV
                logits, _, viz_data = self.forward(
                    next_token_id.unsqueeze(0),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = viz_data["past_key_values"]

        self.train()
        return generated_tokens.cpu().tolist()

    def generate_with_gating(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        gating_policy: Any,
        modulator: Any,
        base_temperature: float = 1.0,
        top_k: int = 40,
        return_trajectory: bool = False,
    ) -> Tuple[List[int], Optional[Dict]]:
        """
        Generate text with RL-controlled gating modulation.

        Args:
            prompt_ids: Initial prompt token IDs (1D tensor)
            max_new_tokens: Maximum number of tokens to generate
            gating_policy: RL policy with get_action(observation) method
            modulator: GatingModulator instance
            base_temperature: Base temperature before gating modulation
            top_k: Keep only top k tokens for sampling
            return_trajectory: If True, return trajectory data for training

        Returns:
            generated_tokens: List of generated token IDs
            trajectory: Dict with observations, actions, rewards (if return_trajectory=True)
        """
        self.eval()
        generated_tokens = prompt_ids.to(self.device).view(-1)

        trajectory = {
            "observations": [],
            "actions": [],
            "logits_history": [],
            "tokens": [],
            "hidden_states": [],
        } if return_trajectory else None

        with torch.no_grad():
            for step in range(max_new_tokens):
                context = generated_tokens[-self.max_context_length:]
                input_ids = context.unsqueeze(0)
                logits, hidden_states = self.forward_with_hidden(input_ids)

                observation = self._build_rl_observation(
                    generated_tokens, hidden_states, logits, step, max_new_tokens,
                )

                if hasattr(gating_policy, "get_action"):
                    gate_signals = gating_policy.get_action(observation)
                else:
                    gate_signals = gating_policy(observation)

                effects = modulator(gate_signals, len(context), self.device)

                next_token_logits = logits[0, -1, :].clone()

                next_token_logits = modulator.apply_repetition_penalty_to_logits(
                    next_token_logits, generated_tokens, effects.repetition_penalty,
                )

                next_token_logits = next_token_logits / effects.temperature
                probs = F.softmax(next_token_logits, dim=-1)

                top_k_val = min(top_k, self.vocab_size)
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k_val)
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[next_token_idx]

                if trajectory is not None:
                    trajectory["observations"].append(observation.clone())
                    trajectory["actions"].append(
                        gate_signals.clone() if isinstance(gate_signals, torch.Tensor)
                        else torch.tensor(gate_signals)
                    )
                    trajectory["logits_history"].append(logits[0, -1, :].clone())
                    trajectory["tokens"].append(next_token_id.item())
                    trajectory["hidden_states"].append(hidden_states[0, -1, :].clone())

                generated_tokens = torch.cat([generated_tokens, next_token_id])

        self.train()
        return generated_tokens.cpu().tolist(), trajectory

    def _build_rl_observation(
        self,
        generated_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        step: int,
        max_steps: int,
    ) -> torch.Tensor:
        """
        Build 64D observation vector for RL policy.

        Layout:
        - Token statistics (10D): repetition ratio, entropy, n-gram diversity
        - Hidden state summary (48D): chunked mean/std pooling
        - Context (6D): step progress, base temperature, previous gate activations (placeholder)
        """
        device = hidden_states.device
        obs_parts = []

        tokens = generated_tokens.cpu().tolist()
        num_tokens = len(tokens)

        if num_tokens > 0:
            unique_ratio = len(set(tokens)) / num_tokens
            repetition_ratio = 1.0 - unique_ratio
        else:
            repetition_ratio = 0.0

        probs = F.softmax(logits[0, -1, :], dim=-1)
        token_entropy = -(probs * (probs + 1e-10).log()).sum().item()
        token_entropy = token_entropy / 10.0

        if num_tokens > 1:
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(num_tokens - 1)]
            bigram_diversity = len(set(bigrams)) / len(bigrams)
        else:
            bigram_diversity = 1.0

        if num_tokens > 2:
            trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(num_tokens - 2)]
            trigram_diversity = len(set(trigrams)) / len(trigrams)
        else:
            trigram_diversity = 1.0

        recent_tokens = tokens[-10:] if len(tokens) >= 10 else tokens
        if len(recent_tokens) > 0:
            recent_unique_ratio = len(set(recent_tokens)) / len(recent_tokens)
            counter = Counter(recent_tokens)
            most_common_freq = counter.most_common(1)[0][1] / len(recent_tokens)
        else:
            recent_unique_ratio = 1.0
            most_common_freq = 0.0

        token_stats = torch.tensor([
            repetition_ratio, token_entropy, bigram_diversity, trigram_diversity,
            recent_unique_ratio, most_common_freq,
            min(num_tokens / 100.0, 1.0),
            0.0, 0.0, 0.0,
        ], device=device)
        obs_parts.append(token_stats)

        last_hidden = hidden_states[0, -1, :]
        chunk_size = self.embed_dim // 24
        hidden_mean = torch.zeros(24, device=device)
        for i in range(24):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            hidden_mean[i] = last_hidden[start_idx:end_idx].mean()

        hidden_std = torch.zeros(24, device=device)
        for i in range(24):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            hidden_std[i] = last_hidden[start_idx:end_idx].std()

        obs_parts.append(hidden_mean)
        obs_parts.append(hidden_std)

        step_progress = step / max(max_steps, 1)
        context_features = torch.tensor([
            step_progress, 1.0,
            0.5, 0.5, 0.5,  # Previous gate activations (placeholder, filled by env)
            0.0,
        ], device=device)
        obs_parts.append(context_features)

        observation = torch.cat(obs_parts)

        if observation.shape[0] < 64:
            padding = torch.zeros(64 - observation.shape[0], device=device)
            observation = torch.cat([observation, padding])
        elif observation.shape[0] > 64:
            observation = observation[:64]

        return observation


def get_model_state_dict(model):
    """Get state_dict, handling torch.compile wrapper transparently."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    """Load state_dict, handling torch.compile wrapper transparently."""
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state_dict)
