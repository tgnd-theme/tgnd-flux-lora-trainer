"""
RunPod pod training script: Per-escort DreamBooth LoRA on Flux 2 Dev.

Downloads escort photos from ZIP URL, trains a face+body LoRA,
saves output to network volume, fires callback webhook.

Environment variables (set by PHP):
  TRAINING_ZIP_URL  — URL to ZIP with training photos
  TRIGGER_WORD      — e.g. "escort_dani" (unique per escort)
  TRAINING_STEPS    — default 1000
  LORA_RANK         — default 16
  RESOLUTION        — default 512 (safe for 24GB GPUs)
  HF_TOKEN          — for gated model download
  CALLBACK_URL      — webhook URL for completion notification
  LORA_ID           — DB record ID (passed back in callback)
  NETWORK_VOLUME    — /runpod-volume (optional, for model caching)
  WEBHOOK_SECRET    — secret token for callback auth
"""

import os
import sys
import time
import subprocess
import zipfile
import shutil
import json


def run(cmd, **kwargs):
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def download_file(url, dest):
    """Download a file using wget."""
    run(f"wget -q --show-progress -O '{dest}' '{url}'")


def validate_images(image_dir):
    """Validate training images. Returns list of valid image paths."""
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    images = []

    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_ext) and not f.startswith('.'):
            path = os.path.join(image_dir, f)
            if os.path.getsize(path) > 10_000:  # > 10KB
                images.append(path)

    return images


def detect_gpu():
    """Detect GPU capabilities for quantization strategy."""
    import torch
    cc = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    props = torch.cuda.get_device_properties(0)
    vram_gb = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / 1024**3
    use_fp8 = cc[0] > 8 or (cc[0] == 8 and cc[1] >= 9)

    print(f"[TRAIN] GPU: {gpu_name}", flush=True)
    print(f"[TRAIN] VRAM: {vram_gb:.1f}GB", flush=True)
    print(f"[TRAIN] Compute: {cc[0]}.{cc[1]}, FP8: {use_fp8}", flush=True)

    return {
        'name': gpu_name,
        'vram_gb': vram_gb,
        'compute': f"{cc[0]}.{cc[1]}",
        'use_fp8': use_fp8,
    }


def fire_callback(callback_url, payload):
    """POST callback to WordPress webhook endpoint."""
    if not callback_url:
        print("[TRAIN] No callback URL, skipping webhook", flush=True)
        return

    try:
        import requests
        print(f"[TRAIN] Firing callback to {callback_url}", flush=True)
        resp = requests.post(
            callback_url,
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'},
        )
        print(f"[TRAIN] Callback response: {resp.status_code} {resp.text[:200]}", flush=True)
    except Exception as e:
        print(f"[TRAIN] Callback failed: {e}", flush=True)


def train(zip_url, trigger_word='escort_person', training_steps=1000,
          lora_rank=16, resolution=512, hf_token='', lora_id='',
          network_volume='/runpod-volume'):
    """
    Run DreamBooth LoRA training. Returns result dict on success, raises on failure.

    Returns:
        dict with keys: status, storage_key, trigger_word, training_time_seconds,
        lora_size_mb, image_count, gpu, resolution
    """
    t_start = time.time()

    print("=" * 60, flush=True)
    print("[TRAIN] TGND Escort LoRA Training", flush=True)
    print(f"[TRAIN] Trigger: {trigger_word}", flush=True)
    print(f"[TRAIN] Steps: {training_steps}, Rank: {lora_rank}, Res: {resolution}", flush=True)
    print(f"[TRAIN] LoRA ID: {lora_id}", flush=True)
    print("=" * 60, flush=True)

    if not zip_url:
        raise ValueError('No training ZIP URL provided')

    # ─── HuggingFace auth ───
    if hf_token:
        run(f"hf auth login --token {hf_token}")

    # ─── Download + extract training images ───
    print("[TRAIN] Downloading training images...", flush=True)
    os.makedirs("/data/images", exist_ok=True)
    zip_path = "/data/training.zip"

    download_file(zip_url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/data/images")

    # Flatten if there's a subdirectory
    for item in os.listdir("/data/images"):
        sub = os.path.join("/data/images", item)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                src = os.path.join(sub, f)
                dst = os.path.join("/data/images", f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(sub, ignore_errors=True)

    os.remove(zip_path)

    # ─── Validate images ───
    images = validate_images("/data/images")
    count = len(images)
    print(f"[TRAIN] Found {count} valid training images", flush=True)

    if count < 5:
        raise ValueError(f'Too few images: {count}')

    if count > 30:
        print(f"[TRAIN] WARNING: {count} images, using first 30", flush=True)
        for img in sorted(images)[30:]:
            os.remove(img)

    # ─── Detect GPU ───
    gpu = detect_gpu()

    # Adjust resolution for VRAM
    if gpu['vram_gb'] >= 40 and resolution < 1024:
        resolution = 1024
        print(f"[TRAIN] Upgraded resolution to 1024 (>40GB VRAM)", flush=True)
    elif gpu['vram_gb'] < 24:
        resolution = 512
        print(f"[TRAIN] Capped resolution at 512 (<24GB VRAM)", flush=True)

    # Quantization config
    bnb_config = "/data/bnb_config.json"
    with open(bnb_config, "w") as f:
        f.write('{"load_in_4bit": true, "bnb_4bit_quant_type": "nf4"}')

    quant_flags = "--do_fp8_training" if gpu['use_fp8'] else f"--bnb_quantization_config_path={bnb_config}"

    # ─── Check for cached base model ───
    model_id = "black-forest-labs/FLUX.1-dev"
    volume_model_path = os.path.join(network_volume, "flux-dev")

    if os.path.exists(volume_model_path) and os.listdir(volume_model_path):
        print(f"[TRAIN] Using cached model from {volume_model_path}", flush=True)
        model_source = volume_model_path
    else:
        print(f"[TRAIN] Will download model from HF Hub: {model_id}", flush=True)
        model_source = model_id

    # ─── Find DreamBooth script ───
    train_script = "/app/diffusers/examples/dreambooth/train_dreambooth_lora_flux2.py"
    if not os.path.exists(train_script):
        train_script_v1 = "/app/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py"
        if os.path.exists(train_script_v1):
            print(f"[TRAIN] Flux 2 script not found, falling back to Flux 1 script", flush=True)
            train_script = train_script_v1
        else:
            raise RuntimeError('No DreamBooth training script found')

    print(f"[TRAIN] Using training script: {train_script}", flush=True)

    # ─── Output directory ───
    output_dir = "/output/escort-lora"
    os.makedirs(output_dir, exist_ok=True)

    # ─── Instance prompt ───
    instance_prompt = f"photo of {trigger_word}"

    # ─── Run DreamBooth training ───
    print("\n" + "=" * 60, flush=True)
    print(f"[TRAIN] Starting DreamBooth LoRA training", flush=True)
    print(f"[TRAIN] Prompt: {instance_prompt}", flush=True)
    print(f"[TRAIN] Steps: {training_steps}, Rank: {lora_rank}, Res: {resolution}", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    grad_accum = 4
    checkpoint_steps = min(250, training_steps // 2)

    train_cmd = f"""accelerate launch {train_script} \
  --pretrained_model_name_or_path={model_source} \
  --instance_data_dir=/data/images \
  --output_dir={output_dir} \
  --instance_prompt="{instance_prompt}" \
  --resolution={resolution} \
  --rank={lora_rank} \
  --lora_alpha={lora_rank} \
  --train_batch_size=1 \
  --gradient_accumulation_steps={grad_accum} \
  --gradient_checkpointing \
  {quant_flags} \
  --remote_text_encoder \
  --cache_latents \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=50 \
  --max_train_steps={training_steps} \
  --checkpointing_steps={checkpoint_steps} \
  --mixed_precision=bf16 \
  --seed=42"""

    run(train_cmd)

    train_elapsed = time.time() - t0
    print(f"[TRAIN] Training completed in {train_elapsed / 60:.1f} minutes", flush=True)

    # ─── Verify output ───
    lora_file = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_file):
        raise RuntimeError('LoRA weights file not produced')

    lora_size_mb = os.path.getsize(lora_file) / 1024 / 1024
    print(f"[TRAIN] LoRA weights: {lora_size_mb:.1f}MB", flush=True)

    # ─── Copy to network volume ───
    storage_key = ""
    volume_lora_dir = os.path.join(network_volume, "loras")

    if os.path.exists(network_volume):
        os.makedirs(volume_lora_dir, exist_ok=True)
        dest_filename = f"escort_{lora_id}.safetensors"
        dest_path = os.path.join(volume_lora_dir, dest_filename)

        shutil.copy2(lora_file, dest_path)
        storage_key = dest_path

        print(f"[TRAIN] LoRA saved to volume: {storage_key}", flush=True)
    else:
        print("[TRAIN] WARNING: No network volume, LoRA only in /output", flush=True)
        storage_key = lora_file

    total_elapsed = time.time() - t_start

    result = {
        'status': 'ready',
        'storage_key': storage_key,
        'trigger_word': trigger_word,
        'training_time_seconds': int(train_elapsed),
        'lora_size_mb': round(lora_size_mb, 1),
        'image_count': count,
        'gpu': gpu['name'],
        'resolution': resolution,
    }

    print("\n" + "=" * 60, flush=True)
    print(f"[TRAIN] ALL DONE in {total_elapsed / 60:.1f} minutes", flush=True)
    print(f"[TRAIN] LoRA: {storage_key}", flush=True)
    print(f"[TRAIN] Trigger: {trigger_word}", flush=True)
    print("=" * 60, flush=True)

    return result


def main():
    """Thin wrapper: reads env vars and calls train(). Backwards compatible for standalone use."""
    zip_url = os.environ.get('TRAINING_ZIP_URL', '')
    trigger_word = os.environ.get('TRIGGER_WORD', 'escort_person')
    training_steps = int(os.environ.get('TRAINING_STEPS', '1000'))
    lora_rank = int(os.environ.get('LORA_RANK', '16'))
    resolution = int(os.environ.get('RESOLUTION', '512'))
    hf_token = os.environ.get('HF_TOKEN', '')
    callback_url = os.environ.get('CALLBACK_URL', '')
    lora_id = os.environ.get('LORA_ID', '')
    network_volume = os.environ.get('NETWORK_VOLUME', '/runpod-volume')
    webhook_secret = os.environ.get('WEBHOOK_SECRET', '')

    try:
        result = train(
            zip_url=zip_url,
            trigger_word=trigger_word,
            training_steps=training_steps,
            lora_rank=lora_rank,
            resolution=resolution,
            hf_token=hf_token,
            lora_id=lora_id,
            network_volume=network_volume,
        )
        result['lora_id'] = lora_id
        result['secret'] = webhook_secret
        fire_callback(callback_url, result)
    except Exception as e:
        print(f"[TRAIN] FAILED: {e}", flush=True)
        fire_callback(callback_url, {
            'lora_id': lora_id, 'status': 'failed',
            'error': str(e), 'secret': webhook_secret,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
