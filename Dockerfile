FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade torch (base has 2.4, need >=2.5 for Flux 2 DreamBooth)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Training dependencies (all pre-installed, no runtime pip install needed)
RUN pip install --no-cache-dir \
    'diffusers>=0.31.0,<0.39.0' \
    'transformers>=4.44.0,<4.52.0' \
    'accelerate>=0.31.0' \
    'peft>=0.14.0' \
    bitsandbytes \
    safetensors \
    sentencepiece \
    protobuf \
    ftfy \
    huggingface_hub \
    requests \
    Pillow

# Clone diffusers for DreamBooth training script (baked into image)
RUN git clone --depth 1 https://github.com/huggingface/diffusers /app/diffusers && \
    pip install --no-cache-dir -r /app/diffusers/examples/dreambooth/requirements_flux.txt 2>/dev/null || true

# Verify critical imports work at build time
RUN python3 -c "from diffusers import FluxPipeline; from peft import LoraConfig; print('All imports OK')"

# Copy training entry script
COPY train_escort_lora.py /app/train_escort_lora.py

# Pod runs training, then exits (not a serverless handler)
CMD ["python3", "-u", "/app/train_escort_lora.py"]
