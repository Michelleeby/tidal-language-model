FROM python:3.11-slim

WORKDIR /app

ARG PLUGIN=tidal

# Install CPU-only PyTorch first (much smaller than CUDA variant)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining inference dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy entire plugin directory — all .py files land flat in /app,
# configs/ at /app/configs/. Matches inference_server.py import style.
COPY plugins/${PLUGIN}/ .

# Pre-download GPT-2 tokenizer into the image so first request is fast
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=configs/base_config.yaml

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "inference_server:app"]
