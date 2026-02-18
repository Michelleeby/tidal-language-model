"""
test_TrajectoryAnalyzer.py

Unit tests for the TrajectoryAnalyzer module — pure-Python analysis
of gate signal trajectories from RL-gated generation.

Updated for single modulation gate (conservative-to-exploratory axis).
"""

import math
import unittest

from plugins.tidal.TrajectoryAnalyzer import (
    _signal_stats,
    _split_windows,
    bootstrap_ci,
    analyze_single,
    analyze_batch,
    get_sweep_grid,
    analyze_sweep,
)


def _make_trajectory(n_steps, modulation_fn, token_fn=None):
    """Build a synthetic trajectory dict matching the lightweight format.

    Single modulation gate: each action is a 1-element list [modulation].
    """
    actions = []
    tokens = []
    token_texts = []
    effects = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)  # normalised [0, 1]
        m = modulation_fn(t)
        actions.append([m])
        tid = token_fn(t) if token_fn else i
        tokens.append(tid)
        token_texts.append(f"tok_{tid}")
        effects.append({
            "temperature": 0.3 + m * 1.7,
            "repetition_penalty": 1.0 + m * 2.5,
            "top_k": int(5 + m * 95),
            "top_p": 0.7 + m * 0.3,
        })
    return {
        "gateSignals": actions,
        "effects": effects,
        "tokenIds": tokens,
        "tokenTexts": token_texts,
    }


class TestSingleTrajectoryAnalysis(unittest.TestCase):
    """Tests for analyze_single on synthetic trajectories with known stats."""

    def setUp(self):
        # Linear ramp modulation [0 → 1]
        self.n = 100
        self.traj = _make_trajectory(
            self.n,
            modulation_fn=lambda t: t,  # 0→1 linear
        )
        self.result = analyze_single(self.traj)

    def test_returns_dict_with_expected_keys(self):
        expected = {
            "signalStats", "signalEvolution", "crossSignalCorrelations",
            "phases", "tokenSignalAlignment",
        }
        self.assertEqual(set(self.result.keys()), expected)

    def test_signal_stats_structure(self):
        """The single modulation signal has mean, std, min, max, q25, q50, q75."""
        stats = self.result["signalStats"]["modulation"]
        for key in ("mean", "std", "min", "max", "q25", "q50", "q75"):
            self.assertIn(key, stats, f"Missing {key} in modulation")
            self.assertIsInstance(stats[key], float)

    def test_modulation_stats(self):
        """Linear ramp 0→1: mean≈0.5, min≈0, max≈1."""
        stats = self.result["signalStats"]["modulation"]
        self.assertAlmostEqual(stats["mean"], 0.5, places=1)
        self.assertAlmostEqual(stats["min"], 0.0, places=2)
        self.assertAlmostEqual(stats["max"], 1.0, places=2)
        self.assertAlmostEqual(stats["q50"], 0.5, places=1)

    def test_signal_evolution_quartile_windows(self):
        """Evolution splits into 4 quartile windows."""
        evo = self.result["signalEvolution"]
        self.assertIn("modulation", evo)
        self.assertEqual(len(evo["modulation"]), 4)

    def test_modulation_evolution_ascending(self):
        """Modulation mean should increase across quartile windows."""
        evo = self.result["signalEvolution"]["modulation"]
        means = [w["mean"] for w in evo]
        for i in range(len(means) - 1):
            self.assertLess(means[i], means[i + 1])

    def test_cross_signal_correlations_empty(self):
        """With a single gate, cross-signal correlations should be empty."""
        corr = self.result["crossSignalCorrelations"]
        self.assertEqual(len(corr), 0)

    def test_phase_detection(self):
        """Phases are detected via Welch t-test on adjacent windows (|t|>2.0)."""
        phases = self.result["phases"]
        self.assertIsInstance(phases, list)
        # modulation ramp should trigger at least one phase transition
        modulation_phases = [p for p in phases if p["signal"] == "modulation"]
        self.assertGreater(len(modulation_phases), 0)
        for p in modulation_phases:
            self.assertIn("windowBoundary", p)
            self.assertIn("tStatistic", p)
            self.assertGreater(abs(p["tStatistic"]), 2.0)

    def test_constant_modulation_no_phases(self):
        """Constant modulation should have no phase transitions."""
        traj = _make_trajectory(100, modulation_fn=lambda t: 0.5)
        result = analyze_single(traj)
        phases = result["phases"]
        modulation_phases = [p for p in phases if p["signal"] == "modulation"]
        self.assertEqual(len(modulation_phases), 0)

    def test_token_signal_alignment(self):
        """Top/bottom 10% of signal values → token indices."""
        alignment = self.result["tokenSignalAlignment"]
        self.assertIn("modulation", alignment)
        self.assertIn("highTokens", alignment["modulation"])
        self.assertIn("lowTokens", alignment["modulation"])
        self.assertIsInstance(alignment["modulation"]["highTokens"], list)
        self.assertIsInstance(alignment["modulation"]["lowTokens"], list)
        self.assertGreater(len(alignment["modulation"]["highTokens"]), 0)
        self.assertGreater(len(alignment["modulation"]["lowTokens"]), 0)

    def test_modulation_alignment_high_tokens_are_late(self):
        """For linear ramp, top 10% tokens should be from late positions."""
        alignment = self.result["tokenSignalAlignment"]["modulation"]
        threshold = int(self.n * 0.9)
        for idx in alignment["highTokens"]:
            self.assertGreaterEqual(idx["position"], threshold)

    def test_modulation_alignment_low_tokens_are_early(self):
        """For linear ramp, bottom 10% tokens should be from early positions."""
        alignment = self.result["tokenSignalAlignment"]["modulation"]
        threshold = int(self.n * 0.1)
        for idx in alignment["lowTokens"]:
            self.assertLess(idx["position"], threshold)

    def test_empty_trajectory_raises(self):
        """Empty trajectory should raise ValueError."""
        empty = {"gateSignals": [], "effects": [], "tokenIds": [], "tokenTexts": []}
        with self.assertRaises(ValueError):
            analyze_single(empty)


class TestSignalStatsEdgeCases(unittest.TestCase):
    """Tests for _signal_stats and _split_windows with small/empty inputs."""

    def test_signal_stats_empty_list_returns_zeroed_dict(self):
        """_signal_stats([]) must return a valid stats dict, not crash."""
        result = _signal_stats([])
        for key in ("mean", "std", "min", "max", "q25", "q50", "q75"):
            self.assertIn(key, result)
            self.assertEqual(result[key], 0.0)

    def test_signal_stats_single_value(self):
        """_signal_stats([v]) should return that value for all positional stats."""
        result = _signal_stats([0.7])
        self.assertAlmostEqual(result["mean"], 0.7)
        self.assertAlmostEqual(result["min"], 0.7)
        self.assertAlmostEqual(result["max"], 0.7)
        self.assertEqual(result["std"], 0.0)

    def test_split_windows_fewer_than_n(self):
        """_split_windows with fewer elements than windows produces empty sublists."""
        windows = _split_windows([0.1, 0.2], 4)
        self.assertEqual(len(windows), 4)
        # Some windows will be empty
        empty_count = sum(1 for w in windows if len(w) == 0)
        self.assertGreater(empty_count, 0)

    def test_analyze_single_one_step_no_crash(self):
        """analyze_single must not crash on a 1-step trajectory."""
        traj = _make_trajectory(1, modulation_fn=lambda t: 0.5)
        result = analyze_single(traj)
        self.assertIn("signalStats", result)
        self.assertIn("signalEvolution", result)
        # Evolution has 4 windows; some will be zeroed
        self.assertEqual(len(result["signalEvolution"]["modulation"]), 4)

    def test_analyze_single_two_steps_no_crash(self):
        """analyze_single must not crash on a 2-step trajectory."""
        traj = _make_trajectory(2, modulation_fn=lambda t: 0.3)
        result = analyze_single(traj)
        self.assertIn("signalStats", result)
        self.assertIn("signalEvolution", result)

    def test_analyze_single_three_steps_no_crash(self):
        """analyze_single must not crash on a 3-step trajectory."""
        traj = _make_trajectory(3, modulation_fn=lambda t: t)
        result = analyze_single(traj)
        self.assertIn("signalStats", result)
        # Phases and correlations should still have valid structure
        self.assertIsInstance(result["phases"], list)
        self.assertIsInstance(result["crossSignalCorrelations"], dict)


class TestBatchAnalysis(unittest.TestCase):
    """Tests for analyze_batch with two prompts having different signal profiles."""

    def setUp(self):
        # Prompt A: high modulation (0.8)
        self.traj_a = _make_trajectory(50, modulation_fn=lambda t: 0.8)
        # Prompt B: low modulation (0.2)
        self.traj_b = _make_trajectory(50, modulation_fn=lambda t: 0.2)
        self.result = analyze_batch({
            "Once upon a time": [self.traj_a],
            "The scientist observed": [self.traj_b],
        })

    def test_returns_expected_keys(self):
        expected = {"perPromptSummaries", "crossPromptVariance", "strategyCharacterization"}
        self.assertEqual(set(self.result.keys()), expected)

    def test_per_prompt_summaries(self):
        """Each prompt gets a single-trajectory summary."""
        summaries = self.result["perPromptSummaries"]
        self.assertIn("Once upon a time", summaries)
        self.assertIn("The scientist observed", summaries)
        # Each is a full analyze_single result
        for prompt, summary in summaries.items():
            self.assertIn("signalStats", summary)

    def test_cross_prompt_variance(self):
        """Between-prompt variance should exist for modulation signal."""
        variance = self.result["crossPromptVariance"]
        self.assertIn("modulation", variance)
        self.assertIn("betweenPromptVar", variance["modulation"])
        self.assertIn("withinPromptVar", variance["modulation"])

    def test_between_variance_exceeds_within(self):
        """When prompts differ, between-prompt variance > within-prompt variance."""
        variance = self.result["crossPromptVariance"]
        # Modulation: 0.8 vs 0.2 between, 0 within (constant per prompt)
        self.assertGreater(
            variance["modulation"]["betweenPromptVar"],
            variance["modulation"]["withinPromptVar"],
        )

    def test_strategy_characterization(self):
        """Strategy summary includes global means and stds."""
        strategy = self.result["strategyCharacterization"]
        self.assertIn("modulation", strategy)
        self.assertIn("globalMean", strategy["modulation"])
        self.assertIn("globalStd", strategy["modulation"])

    def test_strategy_global_mean_modulation(self):
        """Global mean modulation across both prompts ≈ 0.5."""
        strategy = self.result["strategyCharacterization"]
        self.assertAlmostEqual(strategy["modulation"]["globalMean"], 0.5, places=1)


class TestSweepAnalysis(unittest.TestCase):
    """Tests for get_sweep_grid and analyze_sweep with single modulation gate."""

    def test_sweep_grid_has_3_configs(self):
        grid = get_sweep_grid()
        self.assertEqual(len(grid), 3)

    def test_sweep_grid_contains_endpoints_and_midpoint(self):
        """Grid should contain [0.0], [0.5], [1.0]."""
        grid = get_sweep_grid()
        self.assertIn([0.0], grid)
        self.assertIn([0.5], grid)
        self.assertIn([1.0], grid)

    def test_sweep_grid_elements_are_1d(self):
        """Each config should be a 1-element list."""
        grid = get_sweep_grid()
        for cfg in grid:
            self.assertEqual(len(cfg), 1)

    def test_analyze_sweep_returns_expected_keys(self):
        """Sweep analysis returns configComparisons and interpretabilityMap."""
        sweep_data = {}
        for cfg in get_sweep_grid():
            key = f"{cfg[0]:.1f}"
            traj = _make_trajectory(
                20,
                modulation_fn=lambda t, m=cfg[0]: m,
            )
            sweep_data[key] = {
                "trajectory": traj,
                "text": f"Generated text for config {key}",
            }
        result = analyze_sweep(sweep_data)
        self.assertIn("configComparisons", result)
        self.assertIn("interpretabilityMap", result)

    def test_config_comparisons_structure(self):
        """Each config comparison has signal stats and text properties."""
        sweep_data = {}
        for cfg in get_sweep_grid():
            key = f"{cfg[0]:.1f}"
            traj = _make_trajectory(
                20,
                modulation_fn=lambda t, m=cfg[0]: m,
            )
            sweep_data[key] = {
                "trajectory": traj,
                "text": f"The cat sat on the mat. " * 5,
            }
        result = analyze_sweep(sweep_data)
        for key, comp in result["configComparisons"].items():
            self.assertIn("signalStats", comp)
            self.assertIn("textProperties", comp)
            self.assertIn("wordCount", comp["textProperties"])
            self.assertIn("uniqueTokenRatio", comp["textProperties"])

    def test_interpretability_map_per_signal(self):
        """Interpretability map shows marginal effect of the modulation signal."""
        sweep_data = {}
        for cfg in get_sweep_grid():
            key = f"{cfg[0]:.1f}"
            # Use config values to produce different word counts
            word_count = int(10 + cfg[0] * 50)
            text = " ".join([f"word{i}" for i in range(word_count)])
            traj = _make_trajectory(
                20,
                modulation_fn=lambda t, m=cfg[0]: m,
            )
            sweep_data[key] = {"trajectory": traj, "text": text}

        result = analyze_sweep(sweep_data)
        imap = result["interpretabilityMap"]
        self.assertIn("modulation", imap)
        self.assertIn("lowConfig", imap["modulation"])
        self.assertIn("highConfig", imap["modulation"])
        self.assertIn("effect", imap["modulation"])


class TestBootstrapCI(unittest.TestCase):
    """Tests for bootstrap_ci helper function."""

    def test_returns_expected_keys(self):
        """bootstrap_ci returns dict with mean, ci_low, ci_high."""
        result = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
        for key in ("mean", "ci_low", "ci_high"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)

    def test_empty_list_returns_zeros(self):
        """Empty input returns all zeros."""
        result = bootstrap_ci([])
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)

    def test_single_value_degenerate(self):
        """Single value: mean equals value, CI collapses to that value."""
        result = bootstrap_ci([3.14])
        self.assertAlmostEqual(result["mean"], 3.14)
        self.assertAlmostEqual(result["ci_low"], 3.14)
        self.assertAlmostEqual(result["ci_high"], 3.14)

    def test_ci_contains_known_mean(self):
        """For a known distribution, the true mean should be within the CI."""
        import random as _rng
        _rng.seed(99)
        values = [_rng.gauss(5.0, 1.0) for _ in range(200)]
        result = bootstrap_ci(values, seed=42, n_bootstrap=2000)
        self.assertLessEqual(result["ci_low"], 5.0)
        self.assertGreaterEqual(result["ci_high"], 5.0)

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        r1 = bootstrap_ci(values, seed=123)
        r2 = bootstrap_ci(values, seed=123)
        self.assertEqual(r1, r2)

    def test_ci_low_leq_mean_leq_ci_high(self):
        """CI bounds are ordered: ci_low <= mean <= ci_high."""
        values = [1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0]
        result = bootstrap_ci(values, seed=42)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertLessEqual(result["mean"], result["ci_high"])

    def test_percentile_indices_use_n_minus_1(self):
        """Percentile indices should use p*(n-1) convention, matching _quantile."""
        values = list(range(1, 101))  # [1..100], mean = 50.5
        result = bootstrap_ci(values, seed=0, n_bootstrap=100)
        result2 = bootstrap_ci(values, seed=0, n_bootstrap=100)
        self.assertEqual(result, result2)  # deterministic
        import random as _rng
        rng = _rng.Random(0)
        n = len(values)
        boot_means = []
        for _ in range(100):
            sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
            from statistics import mean
            boot_means.append(mean(sample))
        boot_means.sort()
        # With (n-1) convention: hi_idx = floor(0.975 * 99) = 96
        self.assertAlmostEqual(result["ci_high"], boot_means[96])
        self.assertAlmostEqual(result["ci_low"], boot_means[2])


class TestBatchAnalysisWithBootstrap(unittest.TestCase):
    """Tests for analyze_batch with bootstrap parameter."""

    def setUp(self):
        self.traj_a = _make_trajectory(50, modulation_fn=lambda t: 0.8)
        self.traj_b = _make_trajectory(50, modulation_fn=lambda t: 0.2)
        self.prompts = {
            "Once upon a time": [self.traj_a],
            "The scientist observed": [self.traj_b],
        }

    def test_bootstrap_false_omits_ci_keys(self):
        """bootstrap=False (default) produces no CI keys."""
        result = analyze_batch(self.prompts)
        self.assertNotIn("betweenPromptVar_ci", result["crossPromptVariance"]["modulation"])
        self.assertNotIn("globalMean_ci", result["strategyCharacterization"]["modulation"])

    def test_bootstrap_true_adds_ci_keys(self):
        """bootstrap=True adds betweenPromptVar_ci and globalMean_ci."""
        result = analyze_batch(self.prompts, bootstrap=True)
        self.assertIn("betweenPromptVar_ci", result["crossPromptVariance"]["modulation"])
        ci = result["crossPromptVariance"]["modulation"]["betweenPromptVar_ci"]
        for key in ("mean", "ci_low", "ci_high"):
            self.assertIn(key, ci)

        self.assertIn("globalMean_ci", result["strategyCharacterization"]["modulation"])
        ci2 = result["strategyCharacterization"]["modulation"]["globalMean_ci"]
        for key in ("mean", "ci_low", "ci_high"):
            self.assertIn(key, ci2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
