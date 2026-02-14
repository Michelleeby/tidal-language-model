FROM python:3.11-slim

WORKDIR /app

ARG PLUGIN=tidal

# Install CPU-only PyTorch first (much smaller than CUDA variant)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining inference dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Preserve plugins/<PLUGIN>/ package structure so absolute imports
# (e.g. from plugins.tidal.TransformerLM) resolve correctly.
COPY plugins/__init__.py plugins/__init__.py
COPY plugins/${PLUGIN}/ plugins/${PLUGIN}/

# Pre-download GPT-2 tokenizer into the image so first request is fast
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=plugins/tidal/configs/base_config.yaml

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "plugins.tidal.inference_server:app"]
