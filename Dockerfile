FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir \
    datasets==4.5.0 \
    transformers==5.1.0 \
    tensorboard==2.20.0 \
    redis==7.1.1 \
    tqdm==4.67.1

WORKDIR /workspace
