FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade torch + torchaudio + torchvision (must all match)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone ai-toolkit (ostris) — proven FLUX.2 LoRA training framework
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /app/ai-toolkit && \
    cd /app/ai-toolkit && \
    git submodule update --init --recursive

# Install ai-toolkit dependencies
RUN pip install --no-cache-dir -r /app/ai-toolkit/requirements.txt

# Re-install torch stack AFTER requirements.txt to fix version conflicts
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install additional deps for our handler
RUN pip install --no-cache-dir \
    huggingface_hub \
    requests \
    runpod \
    safetensors

# Verify critical imports
RUN python3 -c "import torch; print(f'torch {torch.__version__} CUDA {torch.cuda.is_available()}')"

# Copy training module + serverless handler + entrypoint
COPY train_escort_lora.py /app/train_escort_lora.py
COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Entrypoint: auto-detects serverless (RUNPOD_ENDPOINT_ID) vs pod mode
CMD ["/app/entrypoint.sh"]
