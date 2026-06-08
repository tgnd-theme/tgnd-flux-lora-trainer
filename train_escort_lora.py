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
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        error_detail = result.stderr[-2000:] if result.stderr else "no stderr"
        raise RuntimeError(f"Command failed (exit {result.returncode}):\n{error_detail}")


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
        os.environ['HF_TOKEN'] = hf_token
        run(f"huggingface-cli login --token {hf_token}")

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

    if count > 150:
        print(f"[TRAIN] WARNING: {count} images, using first 150", flush=True)
        for img in sorted(images)[150:]:
            os.remove(img)

    # ─── Detect GPU ───
    gpu = detect_gpu()

    # Adjust resolution for VRAM
    if gpu['vram_gb'] >= 80 and resolution < 1024:
        resolution = 1024
        print(f"[TRAIN] Upgraded resolution to 1024 (>=80GB VRAM)", flush=True)
    elif gpu['vram_gb'] < 48:
        resolution = min(resolution, 512)
        print(f"[TRAIN] Capped resolution at {resolution} (<48GB VRAM)", flush=True)

    # Quantization config for NF4 (saves VRAM)
    bnb_config = "/data/bnb_config.json"
    with open(bnb_config, "w") as f:
        f.write('{"load_in_4bit": true, "bnb_4bit_quant_type": "nf4"}')

    # ─── Base model ───
    # Download to LOCAL disk (fast, no volume I/O issues).
    # Volume is only used for final LoRA output (~20MB).
    model_id = "black-forest-labs/FLUX.2-dev"
    local_model_path = "/data/models/FLUX.2-dev"

    # Check if model already on local disk (warm worker reuse)
    cache_valid = False
    if os.path.exists(os.path.join(local_model_path, "model_index.json")):
        safetensors = []
        for root, dirs, files in os.walk(local_model_path):
            safetensors += [f for f in files if f.endswith('.safetensors')]
        cache_valid = len(safetensors) >= 3
        if not cache_valid:
            print(f"[TRAIN] Local cache incomplete ({len(safetensors)} safetensors), re-downloading...", flush=True)
            shutil.rmtree(local_model_path, ignore_errors=True)

    if cache_valid:
        model_source = local_model_path
        print(f"[TRAIN] Model from local cache: {model_source}", flush=True)
    else:
        print(f"[TRAIN] Downloading {model_id} to local disk...", flush=True)
        from huggingface_hub import snapshot_download
        os.makedirs(local_model_path, exist_ok=True)
        model_source = snapshot_download(
            model_id,
            local_dir=local_model_path,
            token=os.environ.get('HF_TOKEN', hf_token),
            ignore_patterns=["*.onnx", "*.xml"],
        )
        print(f"[TRAIN] Model downloaded to: {model_source}", flush=True)

    # ─── Find DreamBooth script ───
    train_script = "/app/diffusers/examples/dreambooth/train_dreambooth_lora_flux2.py"
    if not os.path.exists(train_script):
        # Fallback to Flux 1 script with Flux 1 model
        train_script = "/app/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py"
        model_source = "black-forest-labs/FLUX.1-dev"
        print(f"[TRAIN] Flux 2 script not found, falling back to Flux 1", flush=True)
        if not os.path.exists(train_script):
            raise RuntimeError('DreamBooth training script not found')

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

    # Clean up previous checkpoints to save disk space
    for item in os.listdir(output_dir) if os.path.exists(output_dir) else []:
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            print(f"[TRAIN] Cleaning old checkpoint: {item}", flush=True)
            shutil.rmtree(item_path, ignore_errors=True)

    # Build training command — args differ between Flux 1 and Flux 2 scripts
    is_flux2 = "flux2" in train_script
    if is_flux2:
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
  --bnb_quantization_config_path={bnb_config} \
  --cache_latents \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=50 \
  --max_train_steps={training_steps} \
  --checkpointing_steps={checkpoint_steps} \
  --resume_from_checkpoint=latest \
  --mixed_precision=bf16 \
  --seed=42"""
    else:
        train_cmd = f"""accelerate launch {train_script} \
  --pretrained_model_name_or_path={model_source} \
  --instance_data_dir=/data/images \
  --output_dir={output_dir} \
  --instance_prompt="{instance_prompt}" \
  --resolution={resolution} \
  --rank={lora_rank} \
  --train_batch_size=1 \
  --gradient_accumulation_steps={grad_accum} \
  --gradient_checkpointing \
  --cache_latents \
  --optimizer=adamw8bit \
  --learning_rate=1e-4 \
  --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=50 \
  --max_train_steps={training_steps} \
  --checkpointing_steps={checkpoint_steps} \
  --resume_from_checkpoint=latest \
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

    # ─── Save LoRA: volume → HuggingFace Hub → callback upload ───
    storage_key = ""
    dest_filename = f"escort_{lora_id}.safetensors"

    # Try 1: Network volume (fastest, if available)
    volume_lora_dir = os.path.join(network_volume, "loras")
    if os.path.exists(network_volume):
        os.makedirs(volume_lora_dir, exist_ok=True)
        dest_path = os.path.join(volume_lora_dir, dest_filename)
        shutil.copy2(lora_file, dest_path)
        storage_key = dest_path
        print(f"[TRAIN] LoRA saved to volume: {storage_key}", flush=True)

    # Try 2: HuggingFace Hub (reliable, works without volume)
    if not storage_key and hf_token:
        try:
            from huggingface_hub import HfApi
            hf_repo = "JulioIglesiass/tgnd-loras"
            api = HfApi(token=hf_token)
            # Create repo if needed (private)
            try:
                api.create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)
            except Exception:
                pass
            api.upload_file(
                path_or_fileobj=lora_file,
                path_in_repo=dest_filename,
                repo_id=hf_repo,
                repo_type="model",
            )
            storage_key = f"hf://{hf_repo}/{dest_filename}"
            print(f"[TRAIN] LoRA uploaded to HF Hub: {storage_key}", flush=True)
        except Exception as e:
            print(f"[TRAIN] HF Hub upload failed: {e}", flush=True)

    # Try 3: Callback URL upload (WordPress)
    callback_url = os.environ.get('CALLBACK_URL', '')
    webhook_secret = os.environ.get('WEBHOOK_SECRET', '')
    if not storage_key and callback_url:
        try:
            import requests
            upload_url = callback_url.replace('/webhook', '/upload-lora')
            print(f"[TRAIN] Uploading LoRA to {upload_url}...", flush=True)
            with open(lora_file, 'rb') as f:
                resp = requests.post(
                    upload_url,
                    files={'lora_file': (dest_filename, f, 'application/octet-stream')},
                    data={'lora_id': lora_id, 'secret': webhook_secret},
                    timeout=300,
                )
            if resp.status_code == 200:
                result_data = resp.json()
                storage_key = result_data.get('storage_key', '')
                print(f"[TRAIN] LoRA uploaded: {storage_key}", flush=True)
            else:
                print(f"[TRAIN] Upload failed: {resp.status_code} {resp.text[:200]}", flush=True)
        except Exception as e:
            print(f"[TRAIN] Upload failed: {e}", flush=True)

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


def serve_output(output_dir, port=19123):
    """Start a simple HTTP file server on the given port to allow downloading output files."""
    import http.server
    import functools

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=output_dir)
    server = http.server.HTTPServer(('0.0.0.0', port), handler)
    print(f"\n[SERVE] HTTP file server running on port {port}", flush=True)
    print(f"[SERVE] Serving files from {output_dir}", flush=True)
    print(f"[SERVE] Download LoRA at: http://0.0.0.0:{port}/pytorch_lora_weights.safetensors", flush=True)
    print("[SERVE] Server will run indefinitely — stop the pod when done downloading.", flush=True)
    server.serve_forever()


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

    # After training completes (success or fail), serve output directory via HTTP
    # so files can be downloaded via the pod's proxy URL
    output_dir = os.environ.get('OUTPUT_DIR', '/output/escort-lora')
    if os.path.isdir(output_dir):
        serve_output(output_dir)


if __name__ == "__main__":
    main()
