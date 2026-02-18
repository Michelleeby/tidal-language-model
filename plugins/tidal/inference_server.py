"""
inference_server.py

Flask HTTP server for model inference, designed to run as a sidecar
container alongside the dashboard. Reuses TransformerLM and Generator
logic directly rather than spawning subprocesses.

Endpoints:
    POST /generate               — run text generation
    POST /analyze-trajectories   — batch trajectory analysis
    GET  /health                 — liveness check

Environment variables:
    CONFIG_PATH  — path to base_config.yaml (default: configs/base_config.yaml)
    PORT         — listen port (default: 5000)
"""

import os
import sys
import time
import logging
from collections import OrderedDict

import torch
from flask import Flask, request, jsonify
from ruamel.yaml import YAML

from plugins.tidal.TransformerLM import TransformerLM
from plugins.tidal.DataPipeline import get_tokenizer

# RL components (optional)
try:
    from plugins.tidal.GatingPolicyAgent import create_agent
    from plugins.tidal.GatingModulator import GatingModulator, FixedGatingPolicy, RandomGatingPolicy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference_server")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Globals initialised at startup
# ---------------------------------------------------------------------------

DEVICE = "cpu"
MAX_CACHED_MODELS = 3

_config = None
_tokenizer = None
_model_cache: OrderedDict[str, TransformerLM] = OrderedDict()


def _get_config():
    global _config
    if _config is None:
        config_path = os.environ.get("CONFIG_PATH", "configs/base_config.yaml")
        yaml = YAML(typ="safe")
        with open(config_path, "r") as f:
            _config = yaml.load(f)
    return _config


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_tokenizer()
    return _tokenizer


def _get_model(checkpoint_path: str) -> TransformerLM:
    """Load a model from checkpoint, using an LRU cache."""
    if checkpoint_path in _model_cache:
        # Move to end (most recently used)
        _model_cache.move_to_end(checkpoint_path)
        return _model_cache[checkpoint_path]

    config = _get_config()
    tokenizer = _get_tokenizer()
    vocab_size = config.get("VOCAB_SIZE", tokenizer.vocab_size)

    logger.info(f"Loading model from {checkpoint_path}")
    model = TransformerLM(vocab_size=vocab_size, config=config)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _model_cache[checkpoint_path] = model
    # Evict oldest if over capacity
    while len(_model_cache) > MAX_CACHED_MODELS:
        evicted_path, _ = _model_cache.popitem(last=False)
        logger.info(f"Evicted model from cache: {evicted_path}")

    return model


def _serialize_trajectory(trajectory, tokenizer):
    """Convert lightweight trajectory dict to JSON-serializable format."""
    gate_signals = []
    for action in trajectory["actions"]:
        if hasattr(action, "tolist"):
            gate_signals.append(action.tolist())
        else:
            gate_signals.append(list(action))

    token_ids = trajectory["tokens"]
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        "gateSignals": gate_signals,
        "effects": trajectory["effects"],
        "tokenIds": token_ids,
        "tokenTexts": token_texts,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)

    checkpoint = data.get("checkpoint")
    if not checkpoint:
        return jsonify({"error": "checkpoint is required"}), 400

    if not os.path.exists(checkpoint):
        return jsonify({"error": f"Checkpoint not found: {checkpoint}"}), 404

    prompt = data.get("prompt", "")
    max_tokens = int(data.get("maxTokens", 50))
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("topK", 50))
    gating_mode = data.get("gatingMode", "none")
    rl_checkpoint = data.get("rlCheckpoint")
    modulation = float(data.get("modulation", 0.5))

    tokenizer = _get_tokenizer()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        return jsonify({"error": "Prompt could not be tokenized"}), 400

    prompt_ids = torch.tensor(encoded, dtype=torch.long)

    try:
        model = _get_model(checkpoint)
    except Exception as e:
        logger.exception("Failed to load model")
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    start = time.time()

    try:
        trajectory = None

        if gating_mode == "learned" and rl_checkpoint and RL_AVAILABLE:
            config = _get_config()
            modulator = GatingModulator(config)

            agent = create_agent(config, DEVICE)
            rl_state = torch.load(rl_checkpoint, map_location=DEVICE)
            agent.load_state_dict(rl_state["agent_state_dict"])
            agent.eval()

            generated_ids, trajectory = model.generate_with_gating(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                gating_policy=agent,
                modulator=modulator,
                base_temperature=temperature,
                top_k=top_k,
                trajectory_mode="lightweight",
            )
        elif gating_mode == "fixed" and RL_AVAILABLE:
            config = _get_config()
            modulator = GatingModulator(config)
            policy = FixedGatingPolicy(
                modulation=modulation, device=DEVICE,
            )

            generated_ids, trajectory = model.generate_with_gating(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                gating_policy=policy,
                modulator=modulator,
                base_temperature=temperature,
                top_k=top_k,
                trajectory_mode="lightweight",
            )
        elif gating_mode == "random" and RL_AVAILABLE:
            config = _get_config()
            modulator = GatingModulator(config)
            policy = RandomGatingPolicy(device=DEVICE)

            generated_ids, trajectory = model.generate_with_gating(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                gating_policy=policy,
                modulator=modulator,
                base_temperature=temperature,
                top_k=top_k,
                trajectory_mode="lightweight",
            )
        else:
            generated_ids = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=1.2,
            )
    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": f"Generation failed: {e}"}), 500

    elapsed_ms = int((time.time() - start) * 1000)
    text = tokenizer.decode(generated_ids)

    result = {
        "text": text,
        "tokensGenerated": len(generated_ids) - len(prompt_ids),
        "elapsedMs": elapsed_ms,
    }

    if trajectory is not None:
        result["trajectory"] = _serialize_trajectory(trajectory, tokenizer)

    return jsonify(result)


@app.route("/analyze-trajectories", methods=["POST"])
def analyze_trajectories():
    from plugins.tidal.TrajectoryAnalyzer import (
        analyze_batch,
        analyze_sweep,
        get_sweep_grid,
    )

    data = request.get_json(force=True)

    checkpoint = data.get("checkpoint")
    if not checkpoint:
        return jsonify({"error": "checkpoint is required"}), 400

    prompts = data.get("prompts")
    if not prompts:
        return jsonify({"error": "prompts is required and must be non-empty"}), 400

    if not os.path.exists(checkpoint):
        return jsonify({"error": f"Checkpoint not found: {checkpoint}"}), 404

    max_tokens = int(data.get("maxTokens", 50))
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("topK", 50))
    gating_mode = data.get("gatingMode", "fixed")
    rl_checkpoint = data.get("rlCheckpoint")
    samples_per_prompt = int(data.get("samplesPerPrompt", 1))
    include_extreme_values = data.get("includeExtremeValues", False)
    bootstrap = data.get("bootstrap", False)

    tokenizer = _get_tokenizer()

    try:
        model = _get_model(checkpoint)
    except Exception as e:
        logger.exception("Failed to load model")
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    def _run_generation(prompt_text, m=0.5):
        """Run a single generation and return (text, serialized_trajectory)."""
        encoded = tokenizer.encode(prompt_text)
        if not encoded:
            return None, None
        prompt_ids = torch.tensor(encoded, dtype=torch.long)
        config = _get_config()

        if gating_mode == "learned" and rl_checkpoint and RL_AVAILABLE:
            modulator = GatingModulator(config)
            agent = create_agent(config, DEVICE)
            rl_state = torch.load(rl_checkpoint, map_location=DEVICE)
            agent.load_state_dict(rl_state["agent_state_dict"])
            agent.eval()
            gen_ids, traj = model.generate_with_gating(
                prompt_ids=prompt_ids, max_new_tokens=max_tokens,
                gating_policy=agent, modulator=modulator,
                base_temperature=temperature, top_k=top_k,
                trajectory_mode="lightweight",
            )
        elif gating_mode == "random" and RL_AVAILABLE:
            modulator = GatingModulator(config)
            policy = RandomGatingPolicy(device=DEVICE)
            gen_ids, traj = model.generate_with_gating(
                prompt_ids=prompt_ids, max_new_tokens=max_tokens,
                gating_policy=policy, modulator=modulator,
                base_temperature=temperature, top_k=top_k,
                trajectory_mode="lightweight",
            )
        elif RL_AVAILABLE:
            modulator = GatingModulator(config)
            policy = FixedGatingPolicy(
                modulation=m, device=DEVICE,
            )
            gen_ids, traj = model.generate_with_gating(
                prompt_ids=prompt_ids, max_new_tokens=max_tokens,
                gating_policy=policy, modulator=modulator,
                base_temperature=temperature, top_k=top_k,
                trajectory_mode="lightweight",
            )
        else:
            return None, None

        text = tokenizer.decode(gen_ids)
        serialized = _serialize_trajectory(traj, tokenizer)
        return text, serialized

    try:
        # --- Batch generation across prompts ---
        prompt_trajectories = {}
        all_trajectories = {}
        for prompt_text in prompts:
            samples = []
            for _ in range(samples_per_prompt):
                text, traj = _run_generation(prompt_text)
                if traj is not None:
                    samples.append(traj)
            prompt_trajectories[prompt_text] = samples
            all_trajectories[prompt_text] = samples

        batch_analysis = analyze_batch(prompt_trajectories, bootstrap=bootstrap)

        result = {
            "batchAnalysis": batch_analysis,
            "trajectories": all_trajectories,
        }

        # --- Optional sweep analysis ---
        if include_extreme_values:
            sweep_data = {}
            sweep_texts = {}
            first_prompt = prompts[0]
            for cfg in get_sweep_grid():
                key = f"{cfg[0]:.1f}"
                text, traj = _run_generation(first_prompt, m=cfg[0])
                if traj is not None:
                    sweep_data[key] = {"trajectory": traj, "text": text}
                    sweep_texts[key] = text

            result["sweepAnalysis"] = analyze_sweep(sweep_data)
            result["sweepTexts"] = sweep_texts

    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# Entrypoint for local dev (gunicorn is used in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting inference server on port {port}")
    # Eagerly load tokenizer so first request isn't slow
    _get_tokenizer()
    app.run(host="0.0.0.0", port=port, debug=False)
