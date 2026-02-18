"""
test_TrajectoryAnalyzer.py

Unit tests for the TrajectoryAnalyzer module — pure-Python analysis
of gate signal trajectories from RL-gated generation.
"""

import math
import unittest

from plugins.tidal.TrajectoryAnalyzer import (
    _signal_stats,
    _split_windows,
    analyze_single,
    analyze_batch,
    get_sweep_grid,
    analyze_sweep,
)


def _make_trajectory(n_steps, creativity_fn, focus_fn, stability_fn, token_fn=None):
    """Build a synthetic trajectory dict matching the lightweight format."""
    actions = []
    tokens = []
    token_texts = []
    effects = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)  # normalised [0, 1]
        c = creativity_fn(t)
        f = focus_fn(t)
        s = stability_fn(t)
        actions.append([c, f, s])
        tid = token_fn(t) if token_fn else i
        tokens.append(tid)
        token_texts.append(f"tok_{tid}")
        effects.append({
            "temperature": 0.3 + c * 1.7,
            "repetition_penalty": 1.0 + s * 2.5,
            "top_k": int(5 + f * 95),
            "top_p": 0.7 + c * 0.3,
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
        # Linear ramp creativity [0 → 1], constant focus 0.5, inverse stability [1 → 0]
        self.n = 100
        self.traj = _make_trajectory(
            self.n,
            creativity_fn=lambda t: t,          # 0→1 linear
            focus_fn=lambda t: 0.5,             # constant
            stability_fn=lambda t: 1.0 - t,     # 1→0 linear
        )
        self.result = analyze_single(self.traj)

    def test_returns_dict_with_expected_keys(self):
        expected = {
            "signalStats", "signalEvolution", "crossSignalCorrelations",
            "phases", "tokenSignalAlignment",
        }
        self.assertEqual(set(self.result.keys()), expected)

    def test_signal_stats_structure(self):
        """Each signal has mean, std, min, max, q25, q50, q75."""
        for name in ("creativity", "focus", "stability"):
            stats = self.result["signalStats"][name]
            for key in ("mean", "std", "min", "max", "q25", "q50", "q75"):
                self.assertIn(key, stats, f"Missing {key} in {name}")
                self.assertIsInstance(stats[key], float)

    def test_creativity_stats(self):
        """Linear ramp 0→1: mean≈0.5, min≈0, max≈1."""
        stats = self.result["signalStats"]["creativity"]
        self.assertAlmostEqual(stats["mean"], 0.5, places=1)
        self.assertAlmostEqual(stats["min"], 0.0, places=2)
        self.assertAlmostEqual(stats["max"], 1.0, places=2)
        self.assertAlmostEqual(stats["q50"], 0.5, places=1)

    def test_focus_stats_constant(self):
        """Constant 0.5: std=0, q25=q50=q75=0.5."""
        stats = self.result["signalStats"]["focus"]
        self.assertAlmostEqual(stats["mean"], 0.5, places=5)
        self.assertAlmostEqual(stats["std"], 0.0, places=5)
        self.assertAlmostEqual(stats["q25"], 0.5, places=5)
        self.assertAlmostEqual(stats["q75"], 0.5, places=5)

    def test_stability_stats(self):
        """Inverse ramp 1→0: mean≈0.5, min≈0, max≈1."""
        stats = self.result["signalStats"]["stability"]
        self.assertAlmostEqual(stats["mean"], 0.5, places=1)
        self.assertAlmostEqual(stats["min"], 0.0, places=2)
        self.assertAlmostEqual(stats["max"], 1.0, places=2)

    def test_signal_evolution_quartile_windows(self):
        """Evolution splits into 4 quartile windows."""
        evo = self.result["signalEvolution"]
        for name in ("creativity", "focus", "stability"):
            self.assertIn(name, evo)
            self.assertEqual(len(evo[name]), 4)

    def test_creativity_evolution_ascending(self):
        """Creativity mean should increase across quartile windows."""
        evo = self.result["signalEvolution"]["creativity"]
        means = [w["mean"] for w in evo]
        for i in range(len(means) - 1):
            self.assertLess(means[i], means[i + 1])

    def test_stability_evolution_descending(self):
        """Stability mean should decrease across quartile windows."""
        evo = self.result["signalEvolution"]["stability"]
        means = [w["mean"] for w in evo]
        for i in range(len(means) - 1):
            self.assertGreater(means[i], means[i + 1])

    def test_focus_evolution_flat(self):
        """Focus is constant — all quartile means should be 0.5."""
        evo = self.result["signalEvolution"]["focus"]
        for w in evo:
            self.assertAlmostEqual(w["mean"], 0.5, places=2)

    def test_cross_signal_correlations(self):
        """creativity vs constant focus ≈ 0, creativity vs inverse stability ≈ -1."""
        corr = self.result["crossSignalCorrelations"]
        # creativity vs focus: r ≈ 0 (one is constant → correlation undefined/0)
        self.assertAlmostEqual(corr["creativity_focus"], 0.0, places=1)
        # creativity vs stability: perfect inverse
        self.assertAlmostEqual(corr["creativity_stability"], -1.0, places=2)

    def test_phase_detection(self):
        """Phases are detected via Welch t-test on adjacent windows (|t|>2.0)."""
        phases = self.result["phases"]
        self.assertIsInstance(phases, list)
        # creativity ramp should trigger at least one phase transition
        creativity_phases = [p for p in phases if p["signal"] == "creativity"]
        self.assertGreater(len(creativity_phases), 0)
        for p in creativity_phases:
            self.assertIn("windowBoundary", p)
            self.assertIn("tStatistic", p)
            self.assertGreater(abs(p["tStatistic"]), 2.0)

    def test_focus_no_phases(self):
        """Constant focus should have no phase transitions."""
        phases = self.result["phases"]
        focus_phases = [p for p in phases if p["signal"] == "focus"]
        self.assertEqual(len(focus_phases), 0)

    def test_token_signal_alignment(self):
        """Top/bottom 10% of signal values → token indices."""
        alignment = self.result["tokenSignalAlignment"]
        for name in ("creativity", "focus", "stability"):
            self.assertIn(name, alignment)
            self.assertIn("highTokens", alignment[name])
            self.assertIn("lowTokens", alignment[name])
            # High tokens: top 10% of signal values
            self.assertIsInstance(alignment[name]["highTokens"], list)
            self.assertIsInstance(alignment[name]["lowTokens"], list)
            self.assertGreater(len(alignment[name]["highTokens"]), 0)
            self.assertGreater(len(alignment[name]["lowTokens"]), 0)

    def test_creativity_alignment_high_tokens_are_late(self):
        """For linear ramp, top 10% tokens should be from late positions."""
        alignment = self.result["tokenSignalAlignment"]["creativity"]
        threshold = int(self.n * 0.9)
        for idx in alignment["highTokens"]:
            self.assertGreaterEqual(idx["position"], threshold)

    def test_creativity_alignment_low_tokens_are_early(self):
        """For linear ramp, bottom 10% tokens should be from early positions."""
        alignment = self.result["tokenSignalAlignment"]["creativity"]
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
        traj = _make_trajectory(
            1,
            creativity_fn=lambda t: 0.5,
            focus_fn=lambda t: 0.5,
            stability_fn=lambda t: 0.5,
        )
        result = analyze_single(traj)
        self.assertIn("signalStats", result)
        self.assertIn("signalEvolution", result)
        # Evolution has 4 windows; some will be zeroed
        for name in ("creativity", "focus", "stability"):
            self.assertEqual(len(result["signalEvolution"][name]), 4)

    def test_analyze_single_two_steps_no_crash(self):
        """analyze_single must not crash on a 2-step trajectory."""
        traj = _make_trajectory(
            2,
            creativity_fn=lambda t: 0.3,
            focus_fn=lambda t: 0.6,
            stability_fn=lambda t: 0.9,
        )
        result = analyze_single(traj)
        self.assertIn("signalStats", result)
        self.assertIn("signalEvolution", result)

    def test_analyze_single_three_steps_no_crash(self):
        """analyze_single must not crash on a 3-step trajectory."""
        traj = _make_trajectory(
            3,
            creativity_fn=lambda t: t,
            focus_fn=lambda t: 0.5,
            stability_fn=lambda t: 1.0 - t,
        )
        result = analyze_single(traj)
        self.assertIn("signalStats", result)
        # Phases and correlations should still have valid structure
        self.assertIsInstance(result["phases"], list)
        self.assertIsInstance(result["crossSignalCorrelations"], dict)


class TestBatchAnalysis(unittest.TestCase):
    """Tests for analyze_batch with two prompts having different signal profiles."""

    def setUp(self):
        # Prompt A: high creativity (0.8), low focus (0.2), mid stability (0.5)
        self.traj_a = _make_trajectory(
            50,
            creativity_fn=lambda t: 0.8,
            focus_fn=lambda t: 0.2,
            stability_fn=lambda t: 0.5,
        )
        # Prompt B: low creativity (0.2), high focus (0.8), high stability (0.9)
        self.traj_b = _make_trajectory(
            50,
            creativity_fn=lambda t: 0.2,
            focus_fn=lambda t: 0.8,
            stability_fn=lambda t: 0.9,
        )
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
        """Between-prompt variance should exist for each signal."""
        variance = self.result["crossPromptVariance"]
        for name in ("creativity", "focus", "stability"):
            self.assertIn(name, variance)
            self.assertIn("betweenPromptVar", variance[name])
            self.assertIn("withinPromptVar", variance[name])

    def test_between_variance_exceeds_within(self):
        """When prompts differ, between-prompt variance > within-prompt variance."""
        variance = self.result["crossPromptVariance"]
        # Creativity: 0.8 vs 0.2 between, 0 within (constant per prompt)
        self.assertGreater(
            variance["creativity"]["betweenPromptVar"],
            variance["creativity"]["withinPromptVar"],
        )

    def test_strategy_characterization(self):
        """Strategy summary includes global means and stds."""
        strategy = self.result["strategyCharacterization"]
        for name in ("creativity", "focus", "stability"):
            self.assertIn(name, strategy)
            self.assertIn("globalMean", strategy[name])
            self.assertIn("globalStd", strategy[name])

    def test_strategy_global_mean_creativity(self):
        """Global mean creativity across both prompts ≈ 0.5."""
        strategy = self.result["strategyCharacterization"]
        self.assertAlmostEqual(strategy["creativity"]["globalMean"], 0.5, places=1)


class TestSweepAnalysis(unittest.TestCase):
    """Tests for get_sweep_grid and analyze_sweep."""

    def test_sweep_grid_has_15_configs(self):
        grid = get_sweep_grid()
        self.assertEqual(len(grid), 15)

    def test_sweep_grid_contains_corners(self):
        """All 8 corners of [0,1]^3 should be present."""
        grid = get_sweep_grid()
        corners = [
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
        ]
        for corner in corners:
            self.assertIn(corner, grid, f"Missing corner {corner}")

    def test_sweep_grid_contains_axis_configs(self):
        """6 axis-isolated configs: [x, 0.5, 0.5] etc."""
        grid = get_sweep_grid()
        axis_configs = [
            [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],  # creativity axis
            [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],    # focus axis
            [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],    # stability axis
        ]
        for cfg in axis_configs:
            self.assertIn(cfg, grid, f"Missing axis config {cfg}")

    def test_sweep_grid_contains_neutral(self):
        grid = get_sweep_grid()
        self.assertIn([0.5, 0.5, 0.5], grid)

    def test_analyze_sweep_returns_expected_keys(self):
        """Sweep analysis returns configComparisons and interpretabilityMap."""
        # Build minimal sweep data
        sweep_data = {}
        for cfg in get_sweep_grid():
            key = f"{cfg[0]:.1f}_{cfg[1]:.1f}_{cfg[2]:.1f}"
            traj = _make_trajectory(
                20,
                creativity_fn=lambda t, c=cfg[0]: c,
                focus_fn=lambda t, f=cfg[1]: f,
                stability_fn=lambda t, s=cfg[2]: s,
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
            key = f"{cfg[0]:.1f}_{cfg[1]:.1f}_{cfg[2]:.1f}"
            traj = _make_trajectory(
                20,
                creativity_fn=lambda t, c=cfg[0]: c,
                focus_fn=lambda t, f=cfg[1]: f,
                stability_fn=lambda t, s=cfg[2]: s,
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
        """Interpretability map shows marginal effect of each signal dimension."""
        sweep_data = {}
        for cfg in get_sweep_grid():
            key = f"{cfg[0]:.1f}_{cfg[1]:.1f}_{cfg[2]:.1f}"
            # Use config values to produce different word counts
            word_count = int(10 + cfg[0] * 50)
            text = " ".join([f"word{i}" for i in range(word_count)])
            traj = _make_trajectory(
                20,
                creativity_fn=lambda t, c=cfg[0]: c,
                focus_fn=lambda t, f=cfg[1]: f,
                stability_fn=lambda t, s=cfg[2]: s,
            )
            sweep_data[key] = {"trajectory": traj, "text": text}

        result = analyze_sweep(sweep_data)
        imap = result["interpretabilityMap"]
        for name in ("creativity", "focus", "stability"):
            self.assertIn(name, imap)
            self.assertIn("lowConfig", imap[name])
            self.assertIn("highConfig", imap[name])
            self.assertIn("effect", imap[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
