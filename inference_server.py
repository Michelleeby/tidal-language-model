"""
inference_server.py

Flask HTTP server for model inference, designed to run as a sidecar
container alongside the dashboard. Reuses TransformerLM and Generator
logic directly rather than spawning subprocesses.

Endpoints:
    POST /generate  — run text generation
    GET  /health    — liveness check

Environment variables:
    CONFIG_PATH  — path to base_config.yaml (default: configs/base_config.yaml)
    PORT         — listen port (default: 5000)
"""

import os
import time
import logging
from collections import OrderedDict

import torch
from flask import Flask, request, jsonify
from ruamel.yaml import YAML

from TransformerLM import TransformerLM
from DataPipeline import get_tokenizer

# RL components (optional)
try:
    from GatingPolicyAgent import create_agent
    from GatingModulator import GatingModulator, FixedGatingPolicy
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
        if gating_mode == "learned" and rl_checkpoint and RL_AVAILABLE:
            config = _get_config()
            modulator = GatingModulator(config)

            agent = create_agent(config, DEVICE)
            rl_state = torch.load(rl_checkpoint, map_location=DEVICE)
            agent.load_state_dict(rl_state["agent_state_dict"])
            agent.eval()

            generated_ids, _ = model.generate_with_gating(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                gating_policy=agent,
                modulator=modulator,
                base_temperature=temperature,
                top_k=top_k,
                return_trajectory=False,
            )
        elif gating_mode == "fixed" and RL_AVAILABLE:
            config = _get_config()
            modulator = GatingModulator(config)
            policy = FixedGatingPolicy(device=DEVICE)

            generated_ids, _ = model.generate_with_gating(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                gating_policy=policy,
                modulator=modulator,
                base_temperature=temperature,
                top_k=top_k,
                return_trajectory=False,
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

    return jsonify({
        "text": text,
        "tokensGenerated": len(generated_ids) - len(prompt_ids),
        "elapsedMs": elapsed_ms,
    })


# ---------------------------------------------------------------------------
# Entrypoint for local dev (gunicorn is used in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting inference server on port {port}")
    # Eagerly load tokenizer so first request isn't slow
    _get_tokenizer()
    app.run(host="0.0.0.0", port=port, debug=False)
