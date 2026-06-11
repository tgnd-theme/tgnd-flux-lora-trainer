FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install torch 2.8 (required for ai-toolkit FLUX.2 support)
RUN pip install --no-cache-dir \
    torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Clone ai-toolkit
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /app/ai-toolkit && \
    cd /app/ai-toolkit && \
    git submodule update --init --recursive

# Install ai-toolkit requirements
RUN cd /app/ai-toolkit && pip install --no-cache-dir -r requirements.txt

# Pin torch 2.8 again (requirements.txt may downgrade it)
RUN pip install --no-cache-dir \
    torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Additional deps for our handler
RUN pip install --no-cache-dir \
    huggingface_hub requests runpod safetensors pyyaml

# Verify imports
RUN python3 -c "import torch; print(f'torch {torch.__version__} CUDA {torch.cuda.is_available()}'); from toolkit.job import get_job; print('ai-toolkit OK')"

# Copy training module + serverless handler + entrypoint
COPY train_escort_lora.py /app/train_escort_lora.py
COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
