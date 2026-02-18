"""
TrajectoryAnalyzer.py

Pure-Python analysis of gate signal trajectories from RL-gated generation.
No torch dependency — uses only stdlib (math, statistics).

Functions:
    analyze_single(trajectory)  → per-trajectory statistics and phase detection
    analyze_batch(prompts)      → cross-prompt comparison and strategy characterisation
    get_sweep_grid()            → 15 standard extreme-value configs
    analyze_sweep(sweep_data)   → per-config comparisons and interpretability map
"""

import math
import statistics


# ---------------------------------------------------------------------------
# Signal names in order (matches 3D gate vector indices)
# ---------------------------------------------------------------------------

SIGNAL_NAMES = ("creativity", "focus", "stability")


# ---------------------------------------------------------------------------
# Helper: Pearson correlation for two sequences
# ---------------------------------------------------------------------------

def _pearson(xs, ys):
    """Pearson correlation coefficient. Returns 0.0 if either series is constant."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1)) if n > 1 else 0.0
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Helper: quantile (linear interpolation, matching Excel PERCENTILE.INC)
# ---------------------------------------------------------------------------

def _quantile(sorted_vals, q):
    """Quantile via linear interpolation on a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


# ---------------------------------------------------------------------------
# Helper: Welch's t-test for two independent samples
# ---------------------------------------------------------------------------

def _welch_t(a, b):
    """Return Welch's t-statistic comparing means of two samples.
    Returns 0.0 if either sample has zero variance or length < 2.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    va = statistics.variance(a)
    vb = statistics.variance(b)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0.0:
        return 0.0
    return (ma - mb) / denom


# ---------------------------------------------------------------------------
# Helper: compute per-signal statistics
# ---------------------------------------------------------------------------

def _signal_stats(values):
    """Compute descriptive statistics for a list of floats.

    Returns a zeroed-out stats dict when *values* is empty (e.g. when
    ``_split_windows`` produces an empty window for short trajectories).
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "q25": 0.0, "q50": 0.0, "q75": 0.0}
    s = sorted(values)
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(s[0]),
        "max": float(s[-1]),
        "q25": _quantile(s, 0.25),
        "q50": _quantile(s, 0.50),
        "q75": _quantile(s, 0.75),
    }


# ---------------------------------------------------------------------------
# Helper: split list into n roughly-equal windows
# ---------------------------------------------------------------------------

def _split_windows(values, n_windows=4):
    """Split values into n_windows roughly equal-sized sublists."""
    k = len(values)
    base_size = k // n_windows
    remainder = k % n_windows
    windows = []
    start = 0
    for i in range(n_windows):
        size = base_size + (1 if i < remainder else 0)
        windows.append(values[start: start + size])
        start += size
    return windows


# ---------------------------------------------------------------------------
# Helper: text properties for sweep analysis
# ---------------------------------------------------------------------------

def _text_properties(text):
    """Extract simple properties from generated text."""
    words = text.split()
    word_count = len(words)
    unique_tokens = len(set(words))
    unique_ratio = unique_tokens / word_count if word_count > 0 else 0.0
    return {
        "wordCount": word_count,
        "uniqueTokenRatio": float(unique_ratio),
        "charCount": len(text),
    }


# ---------------------------------------------------------------------------
# Public: analyze_single
# ---------------------------------------------------------------------------

def analyze_single(trajectory):
    """Analyse a single generation trajectory.

    Args:
        trajectory: dict with keys gateSignals, effects, tokenIds, tokenTexts.
            gateSignals is a list of [creativity, focus, stability] per step.

    Returns:
        dict with signalStats, signalEvolution, crossSignalCorrelations,
        phases, tokenSignalAlignment.

    Raises:
        ValueError: if trajectory has zero steps.
    """
    signals = trajectory["gateSignals"]
    n = len(signals)
    if n == 0:
        raise ValueError("Trajectory has zero steps — nothing to analyse.")

    # Extract per-signal series
    series = {
        name: [step[i] for step in signals]
        for i, name in enumerate(SIGNAL_NAMES)
    }

    # --- Signal statistics ---
    signal_stats = {name: _signal_stats(vals) for name, vals in series.items()}

    # --- Signal evolution (4 quartile windows) ---
    signal_evolution = {}
    windowed = {}
    for name, vals in series.items():
        windows = _split_windows(vals, 4)
        windowed[name] = windows
        signal_evolution[name] = [_signal_stats(w) for w in windows]

    # --- Cross-signal correlations ---
    pairs = [
        ("creativity", "focus"),
        ("creativity", "stability"),
        ("focus", "stability"),
    ]
    cross = {}
    for a, b in pairs:
        key = f"{a}_{b}"
        cross[key] = _pearson(series[a], series[b])

    # --- Phase detection (Welch t-test on adjacent windows) ---
    phases = []
    for name, windows in windowed.items():
        for i in range(len(windows) - 1):
            t = _welch_t(windows[i], windows[i + 1])
            if abs(t) > 2.0:
                phases.append({
                    "signal": name,
                    "windowBoundary": f"Q{i + 1}→Q{i + 2}",
                    "tStatistic": float(t),
                })

    # --- Token-signal alignment (top/bottom 10%) ---
    token_ids = trajectory["tokenIds"]
    token_texts = trajectory["tokenTexts"]

    alignment = {}
    for name, vals in series.items():
        indexed = sorted(
            [(i, v) for i, v in enumerate(vals)],
            key=lambda x: x[1],
        )
        k = max(1, int(n * 0.1))
        low_entries = indexed[:k]
        high_entries = indexed[-k:]
        alignment[name] = {
            "highTokens": [
                {"position": i, "tokenId": token_ids[i], "tokenText": token_texts[i], "value": v}
                for i, v in high_entries
            ],
            "lowTokens": [
                {"position": i, "tokenId": token_ids[i], "tokenText": token_texts[i], "value": v}
                for i, v in low_entries
            ],
        }

    return {
        "signalStats": signal_stats,
        "signalEvolution": signal_evolution,
        "crossSignalCorrelations": cross,
        "phases": phases,
        "tokenSignalAlignment": alignment,
    }


# ---------------------------------------------------------------------------
# Public: analyze_batch
# ---------------------------------------------------------------------------

def analyze_batch(prompts):
    """Analyse trajectories across multiple prompts.

    Args:
        prompts: dict mapping prompt text → list of trajectories.

    Returns:
        dict with perPromptSummaries, crossPromptVariance, strategyCharacterization.
    """
    per_prompt = {}
    per_prompt_means = {name: [] for name in SIGNAL_NAMES}

    for prompt, trajectories in prompts.items():
        # Average across multiple samples for the same prompt
        if len(trajectories) == 1:
            summary = analyze_single(trajectories[0])
        else:
            # Merge multiple trajectories: concatenate signals
            merged_signals = []
            merged_effects = []
            merged_token_ids = []
            merged_token_texts = []
            for t in trajectories:
                merged_signals.extend(t["gateSignals"])
                merged_effects.extend(t["effects"])
                merged_token_ids.extend(t["tokenIds"])
                merged_token_texts.extend(t["tokenTexts"])
            merged = {
                "gateSignals": merged_signals,
                "effects": merged_effects,
                "tokenIds": merged_token_ids,
                "tokenTexts": merged_token_texts,
            }
            summary = analyze_single(merged)

        per_prompt[prompt] = summary
        for name in SIGNAL_NAMES:
            per_prompt_means[name].append(summary["signalStats"][name]["mean"])

    # --- Cross-prompt variance ---
    cross_variance = {}
    for name in SIGNAL_NAMES:
        between_means = per_prompt_means[name]
        between_var = statistics.pvariance(between_means) if len(between_means) > 1 else 0.0

        # Within-prompt variance: average of per-prompt stds^2
        within_vars = []
        for prompt, summary in per_prompt.items():
            std = summary["signalStats"][name]["std"]
            within_vars.append(std ** 2)
        within_var = statistics.mean(within_vars) if within_vars else 0.0

        cross_variance[name] = {
            "betweenPromptVar": float(between_var),
            "withinPromptVar": float(within_var),
        }

    # --- Strategy characterisation ---
    strategy = {}
    for name in SIGNAL_NAMES:
        all_means = per_prompt_means[name]
        strategy[name] = {
            "globalMean": float(statistics.mean(all_means)),
            "globalStd": float(statistics.pstdev(all_means)) if len(all_means) > 1 else 0.0,
        }

    return {
        "perPromptSummaries": per_prompt,
        "crossPromptVariance": cross_variance,
        "strategyCharacterization": strategy,
    }


# ---------------------------------------------------------------------------
# Public: get_sweep_grid
# ---------------------------------------------------------------------------

def get_sweep_grid():
    """Return 15 standard gate signal configurations for extreme-value analysis.

    8 corners of [0,1]^3 + 6 axis-isolated extremes + 1 neutral baseline.
    """
    # 8 corners
    corners = [
        [float(c), float(f), float(s)]
        for c in (0.0, 1.0) for f in (0.0, 1.0) for s in (0.0, 1.0)
    ]
    # 6 axis-isolated (vary one dimension, hold others at 0.5)
    axis = [
        [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],  # creativity axis
        [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],  # focus axis
        [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],  # stability axis
    ]
    # 1 neutral baseline
    neutral = [[0.5, 0.5, 0.5]]

    return corners + axis + neutral


# ---------------------------------------------------------------------------
# Public: analyze_sweep
# ---------------------------------------------------------------------------

def analyze_sweep(sweep_data):
    """Analyse results from a gate signal sweep experiment.

    Args:
        sweep_data: dict mapping config_key (e.g. "0.0_0.5_0.5") → {trajectory, text}.

    Returns:
        dict with configComparisons and interpretabilityMap.
    """
    comparisons = {}
    for key, data in sweep_data.items():
        traj_analysis = analyze_single(data["trajectory"])
        comparisons[key] = {
            "signalStats": traj_analysis["signalStats"],
            "textProperties": _text_properties(data["text"]),
        }

    # --- Interpretability map: marginal effect of each dimension ---
    imap = {}
    axis_pairs = {
        "creativity": ("0.0_0.5_0.5", "1.0_0.5_0.5"),
        "focus": ("0.5_0.0_0.5", "0.5_1.0_0.5"),
        "stability": ("0.5_0.5_0.0", "0.5_0.5_1.0"),
    }
    for name, (low_key, high_key) in axis_pairs.items():
        low_data = comparisons.get(low_key, {})
        high_data = comparisons.get(high_key, {})
        effect = {}
        if low_data and high_data:
            low_tp = low_data.get("textProperties", {})
            high_tp = high_data.get("textProperties", {})
            for prop in ("wordCount", "uniqueTokenRatio", "charCount"):
                low_val = low_tp.get(prop, 0)
                high_val = high_tp.get(prop, 0)
                effect[prop] = {
                    "low": low_val,
                    "high": high_val,
                    "delta": high_val - low_val,
                }
        imap[name] = {
            "lowConfig": low_key,
            "highConfig": high_key,
            "effect": effect,
        }

    return {
        "configComparisons": comparisons,
        "interpretabilityMap": imap,
    }
