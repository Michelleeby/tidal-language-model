FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than CUDA variant)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining inference dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy only the Python files needed for inference
COPY TransformerLM.py .
COPY DataPipeline.py .
COPY Generator.py .
COPY GatingPolicyAgent.py .
COPY GatingModulator.py .
COPY GatingEnvironment.py .
COPY inference_server.py .

# Copy config files (tracked in git, needed at runtime)
COPY configs/ configs/

# Pre-download GPT-2 tokenizer into the image so first request is fast
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=configs/base_config.yaml

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "inference_server:app"]
