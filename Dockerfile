FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade torch (base has 2.4, need >=2.5 for Flux 2 DreamBooth)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone diffusers from source first (training scripts require dev version)
RUN git clone --depth 1 https://github.com/huggingface/diffusers /app/diffusers

# Install diffusers from source + other training dependencies
RUN pip install --no-cache-dir \
    /app/diffusers \
    'transformers>=4.44.0,<5.0.0' \
    'accelerate>=0.31.0' \
    'peft>=0.14.0' \
    bitsandbytes \
    safetensors \
    sentencepiece \
    protobuf \
    ftfy \
    huggingface_hub \
    requests \
    Pillow \
    runpod

# Install DreamBooth example requirements
RUN pip install --no-cache-dir -r /app/diffusers/examples/dreambooth/requirements_flux.txt 2>/dev/null || true

# Verify critical imports work at build time
RUN python3 -c "from diffusers import FluxPipeline; from peft import LoraConfig; import diffusers; print(f'diffusers {diffusers.__version__} OK')"

# Copy training module + serverless handler
COPY train_escort_lora.py /app/train_escort_lora.py
COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Smart entrypoint: serverless if RUNPOD_ENDPOINT_ID set, else standalone training
CMD ["/app/entrypoint.sh"]
