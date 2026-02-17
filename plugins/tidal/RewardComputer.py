"""
RewardComputer.py

Multi-component reward function for RL gate signal control.
Provides dense per-step rewards based on:
- Perplexity (language quality)
- Diversity (vocabulary variety)
- Focus (sampling entropy of post-filtered distribution)
- Repetition penalty (avoid repeats)
- Coherence (bigram likelihood)

This enables the RL agent to learn gating policies that
improve generation quality across multiple dimensions.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import Counter
import math


class RewardComputer:
    """
    Computes multi-component rewards for RL-controlled generation.

    Reward formula (per-step):
        reward = (
            -perplexity_weight * perplexity_term +
            +diversity_weight * diversity_bonus +
            -repetition_weight * repetition_penalty +
            +coherence_weight * coherence_bonus
        )

    All components are normalized to roughly [-1, 1] range for stable training.
    """

    def __init__(self, config: dict, vocab_size: int):
        """
        Initialize the RewardComputer.

        Args:
            config: Configuration dictionary with reward weights
            vocab_size: Size of vocabulary for perplexity calculation
        """
        self.config = config
        self.vocab_size = vocab_size

        # Reward component weights
        self.perplexity_weight = config.get("RL_REWARD_PERPLEXITY_WEIGHT", 0.30)
        self.diversity_weight = config.get("RL_REWARD_DIVERSITY_WEIGHT", 0.25)
        self.focus_weight = config.get("RL_REWARD_FOCUS_WEIGHT", 0.15)
        self.repetition_weight = config.get("RL_REWARD_REPETITION_WEIGHT", 0.20)
        self.coherence_weight = config.get("RL_REWARD_COHERENCE_WEIGHT", 0.10)

        # Normalization parameters
        self.perplexity_clip = config.get("RL_PERPLEXITY_CLIP", 100.0)
        self.entropy_target = config.get("RL_ENTROPY_TARGET", 5.0)  # Ideal entropy
        self.sampling_entropy_target = config.get("RL_SAMPLING_ENTROPY_TARGET", 2.5)

        # Bigram statistics for coherence (built from training data)
        self.bigram_counts: Optional[Dict[Tuple[int, int], int]] = None
        self.unigram_counts: Optional[Dict[int, int]] = None
        self.total_bigrams = 0
        self.total_unigrams = 0

        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def build_ngram_statistics(self, token_sequences: List[List[int]]):
        """
        Build bigram and unigram statistics from training data.

        Args:
            token_sequences: List of token ID sequences from training corpus
        """
        self.bigram_counts = Counter()
        self.unigram_counts = Counter()

        for sequence in token_sequences:
            for i, token in enumerate(sequence):
                self.unigram_counts[token] += 1
                if i > 0:
                    bigram = (sequence[i-1], token)
                    self.bigram_counts[bigram] += 1

        self.total_bigrams = sum(self.bigram_counts.values())
        self.total_unigrams = sum(self.unigram_counts.values())

    def compute_perplexity_reward(
        self,
        logits: torch.Tensor,
        token_id: int
    ) -> float:
        """
        Compute perplexity-based reward for a single token prediction.

        Lower perplexity = better language modeling = higher reward.

        Args:
            logits: Logits for next token prediction (vocab_size,)
            token_id: The actually sampled token ID

        Returns:
            Normalized perplexity reward in roughly [-1, 0] range
        """
        # Compute log probability of the sampled token
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_prob = log_probs[token_id].item()

        # Perplexity = exp(-log_prob) for a single token
        perplexity = math.exp(-token_log_prob)

        # Clip perplexity for numerical stability
        perplexity = min(perplexity, self.perplexity_clip)

        # Normalize: low perplexity (1) -> 0, high perplexity (clip) -> -1
        # Using log scale for better gradient properties
        log_ppl = math.log(perplexity)
        log_clip = math.log(self.perplexity_clip)

        # Reward: 0 for perplexity=1, -1 for perplexity=clip
        reward = -log_ppl / log_clip

        return reward

    def compute_diversity_reward(
        self,
        logits: torch.Tensor,
    ) -> float:
        """
        Compute diversity reward from logits-distribution entropy.

        Measures H(softmax(logits)) and rewards proximity to the entropy
        target via a Gaussian. This replaces the previous distinct-n metric
        which was trivially saturated at ~1.0 with BPE tokenization.

        The creativity gate controls temperature which directly affects
        logits entropy, creating a real gradient signal for the RL agent.

        Args:
            logits: Logits for next token prediction (vocab_size,)

        Returns:
            Diversity reward in [0, 1] range, peaking at entropy_target
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()

        # Gaussian reward centered on target entropy
        # bandwidth = target / 2 gives useful gradient across the range
        bandwidth = self.entropy_target * 0.5
        deviation = entropy - self.entropy_target
        reward = math.exp(-0.5 * (deviation / bandwidth) ** 2)

        return reward

    def compute_focus_reward(
        self,
        sampling_entropy: float,
    ) -> float:
        """
        Compute focus reward from post-filtered sampling entropy.

        Measures the entropy of the probability distribution after top-k
        and top-p filtering. This creates a direct gradient path for the
        focus gate: focus → top-k → filtered distribution → sampling entropy.

        Uses the same Gaussian-target pattern as compute_diversity_reward().

        Args:
            sampling_entropy: Entropy of the post-filtered probability distribution

        Returns:
            Focus reward in [0, 1] range, peaking at sampling_entropy_target
        """
        bandwidth = self.sampling_entropy_target * 0.5
        deviation = sampling_entropy - self.sampling_entropy_target
        reward = math.exp(-0.5 * (deviation / bandwidth) ** 2)
        return reward

    def compute_repetition_penalty(
        self,
        generated_tokens: List[int],
        new_token: int,
        window_size: int = 10
    ) -> float:
        """
        Compute penalty for repeating recent tokens.

        Higher penalty for tokens that appear frequently in recent context.

        Args:
            generated_tokens: All generated tokens so far
            new_token: The newly sampled token
            window_size: Number of recent tokens to check for repetition

        Returns:
            Repetition penalty in [0, 1] range (0 = no repetition, 1 = severe)
        """
        if len(generated_tokens) == 0:
            return 0.0

        # Check recent window
        recent = generated_tokens[-window_size:] if len(generated_tokens) > window_size else generated_tokens

        # Count occurrences of new token in recent context
        occurrences = recent.count(new_token)

        # Penalty increases with more occurrences
        # 0 occurrences -> 0, 1 occurrence -> 0.25, 2 -> 0.5, 3+ -> 0.75+
        if occurrences == 0:
            penalty = 0.0
        elif occurrences == 1:
            penalty = 0.25
        elif occurrences == 2:
            penalty = 0.5
        else:
            penalty = min(0.75 + 0.05 * (occurrences - 3), 1.0)

        # Extra penalty for immediate repetition
        if len(generated_tokens) > 0 and generated_tokens[-1] == new_token:
            penalty = min(penalty + 0.3, 1.0)

        return penalty

    def compute_coherence_reward(
        self,
        prev_token: int,
        new_token: int
    ) -> float:
        """
        Compute coherence reward based on bigram likelihood.

        Uses pre-built bigram statistics from training data.
        Higher reward for likely bigrams.

        Args:
            prev_token: Previous token ID
            new_token: New token ID

        Returns:
            Coherence reward in [0, 1] range
        """
        if self.bigram_counts is None or self.total_bigrams == 0:
            return 0.5  # Neutral if no statistics available

        bigram = (prev_token, new_token)

        # Compute bigram probability with smoothing
        bigram_count = self.bigram_counts.get(bigram, 0)
        prev_count = self.unigram_counts.get(prev_token, 0)

        if prev_count == 0:
            # Fallback to unigram probability
            new_count = self.unigram_counts.get(new_token, 0)
            prob = (new_count + 1) / (self.total_unigrams + self.vocab_size)
        else:
            # Add-one smoothing
            prob = (bigram_count + 1) / (prev_count + self.vocab_size)

        # Convert probability to reward via log scale
        # Higher probability -> higher reward
        log_prob = math.log(prob)
        log_uniform = math.log(1.0 / self.vocab_size)

        # Normalize: uniform distribution -> 0, high prob -> 1
        coherence = (log_prob - log_uniform) / (-log_uniform)
        coherence = max(0, min(1, coherence))  # Clip to [0, 1]

        return coherence

    def compute_step_reward(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        new_token: int,
        normalize: bool = True,
        sampling_entropy: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward for a single generation step.

        Args:
            logits: Logits for next token prediction (vocab_size,)
            generated_tokens: All tokens generated before this step
            new_token: The newly sampled token
            normalize: Whether to normalize reward using running statistics
            sampling_entropy: Entropy of post-filtered distribution (for focus reward).
                When provided, the focus reward component is included.

        Returns:
            total_reward: Combined reward value
            components: Dict with individual reward components
        """
        # Compute individual components
        perplexity_reward = self.compute_perplexity_reward(logits, new_token)
        diversity_reward = self.compute_diversity_reward(logits)

        repetition_penalty = self.compute_repetition_penalty(generated_tokens, new_token)

        # Coherence needs previous token
        if len(generated_tokens) > 0:
            coherence_reward = self.compute_coherence_reward(generated_tokens[-1], new_token)
        else:
            coherence_reward = 0.5

        # Combine with weights
        total_reward = (
            -self.perplexity_weight * (-perplexity_reward) +  # perplexity_reward is already negative
            self.diversity_weight * diversity_reward +
            -self.repetition_weight * repetition_penalty +
            self.coherence_weight * coherence_reward
        )

        components = {
            "perplexity": perplexity_reward,
            "diversity": diversity_reward,
            "repetition": -repetition_penalty,
            "coherence": coherence_reward,
            "total_raw": total_reward if not normalize else None
        }

        # Focus reward (only when sampling_entropy is provided)
        if sampling_entropy is not None:
            focus_reward = self.compute_focus_reward(sampling_entropy)
            total_reward += self.focus_weight * focus_reward
            components["focus"] = focus_reward

        # Update running statistics and normalize
        if normalize:
            self._update_statistics(total_reward)
            total_reward = (total_reward - self.reward_mean) / (self.reward_std + 1e-8)

        return total_reward, components

    def compute_episode_rewards(
        self,
        logits_history: List[torch.Tensor],
        tokens: List[int],
        gamma: float = 0.99
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute rewards for an entire episode (generation sequence).

        Args:
            logits_history: List of logits tensors for each step
            tokens: List of generated token IDs
            gamma: Discount factor for returns

        Returns:
            rewards: Tensor of step rewards (num_steps,)
            components: Dict with lists of component values per step
        """
        rewards = []
        components = {
            "perplexity": [],
            "diversity": [],
            "repetition": [],
            "coherence": []
        }

        generated_so_far = []

        for i, (logits, token) in enumerate(zip(logits_history, tokens)):
            reward, step_components = self.compute_step_reward(
                logits,
                generated_so_far,
                token,
                normalize=False  # Normalize at end
            )

            rewards.append(reward)
            for key in components:
                if key in step_components:
                    components[key].append(step_components[key])

            generated_so_far.append(token)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Normalize episode rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        return rewards, components

    def compute_returns(
        self,
        rewards: torch.Tensor,
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute discounted returns from rewards.

        Args:
            rewards: Tensor of step rewards (num_steps,)
            gamma: Discount factor

        Returns:
            returns: Tensor of discounted returns (num_steps,)
        """
        returns = torch.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def _update_statistics(self, reward: float):
        """Update running mean and std for reward normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count

        if self.reward_count > 1:
            # Welford's online algorithm for variance
            delta2 = reward - self.reward_mean
            self.reward_std = math.sqrt(
                (self.reward_std ** 2 * (self.reward_count - 2) + delta * delta2)
                / (self.reward_count - 1)
            )

    def reset_statistics(self):
        """Reset running statistics for new training run."""
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0


class AblationRewardComputer(RewardComputer):
    """
    Reward computer for ablation studies.
    Allows selective disabling of reward components.
    """

    def __init__(
        self,
        config: dict,
        vocab_size: int,
        enable_perplexity: bool = True,
        enable_diversity: bool = True,
        enable_focus: bool = True,
        enable_repetition: bool = True,
        enable_coherence: bool = True
    ):
        super().__init__(config, vocab_size)

        # Override weights based on what's enabled
        if not enable_perplexity:
            self.perplexity_weight = 0.0
        if not enable_diversity:
            self.diversity_weight = 0.0
        if not enable_focus:
            self.focus_weight = 0.0
        if not enable_repetition:
            self.repetition_weight = 0.0
        if not enable_coherence:
            self.coherence_weight = 0.0

        # Renormalize weights to sum to 1
        total_weight = (
            self.perplexity_weight +
            self.diversity_weight +
            self.focus_weight +
            self.repetition_weight +
            self.coherence_weight
        )

        if total_weight > 0:
            self.perplexity_weight /= total_weight
            self.diversity_weight /= total_weight
            self.focus_weight /= total_weight
            self.repetition_weight /= total_weight
            self.coherence_weight /= total_weight
