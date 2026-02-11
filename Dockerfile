FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# System deps: Redis server, Node.js 20, git
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    curl \
    git \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    datasets==4.5.0 \
    transformers==5.1.0 \
    tensorboard==2.20.0 \
    redis==7.1.1 \
    tqdm==4.67.1 \
    ruamel.yaml \
    numpy \
    matplotlib \
    scipy \
    scikit-learn

WORKDIR /workspace
