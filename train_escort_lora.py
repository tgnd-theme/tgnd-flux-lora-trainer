"""
RunPod training script: Per-escort LoRA on FLUX.2-dev via ai-toolkit.

Downloads escort photos from ZIP URL, generates per-image captions,
trains a face+body LoRA using ai-toolkit (ostris), saves output to
network volume + HuggingFace Hub.
"""

import os
import sys
import time
import subprocess
import zipfile
import shutil
import json
import yaml


def run(cmd, stream=False, **kwargs):
    print(f"\n>>> {cmd}", flush=True)
    if stream:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
        last_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last_lines.append(line)
            if len(last_lines) > 50:
                last_lines.pop(0)
        proc.wait()
        if proc.returncode != 0:
            tail = "".join(last_lines)[-2000:]
            raise RuntimeError(f"Command failed (exit {proc.returncode}):\n{tail}")
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        if result.returncode != 0:
            error_detail = result.stderr[-2000:] if result.stderr else "no stderr"
            raise RuntimeError(f"Command failed (exit {result.returncode}):\n{error_detail}")


def download_file(url, dest):
    run(f"wget -q --show-progress -O '{dest}' '{url}'")


def validate_images(image_dir):
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    images = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_ext) and not f.startswith('.'):
            path = os.path.join(image_dir, f)
            if os.path.getsize(path) > 10_000:
                images.append(path)
    return images


def generate_captions(image_dir, trigger_word):
    """Generate .txt caption files for each image. Just the trigger word."""
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    count = 0
    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_ext) and not f.startswith('.'):
            stem = os.path.splitext(f)[0]
            caption_path = os.path.join(image_dir, f"{stem}.txt")
            with open(caption_path, "w") as cf:
                cf.write(trigger_word)
            count += 1
    print(f"[TRAIN] Generated {count} caption files (trigger: {trigger_word})", flush=True)
    return count


def detect_gpu():
    import torch
    cc = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    props = torch.cuda.get_device_properties(0)
    vram_gb = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / 1024**3
    print(f"[TRAIN] GPU: {gpu_name} ({vram_gb:.1f}GB VRAM, compute {cc[0]}.{cc[1]})", flush=True)
    return {'name': gpu_name, 'vram_gb': vram_gb, 'compute': f"{cc[0]}.{cc[1]}"}


def build_training_config(trigger_word, image_dir, output_dir, model_id,
                          training_steps, lora_rank, resolution):
    """Build ai-toolkit YAML config."""
    config = {
        'job': 'extension',
        'config': {
            'name': f'escort_{trigger_word}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': output_dir,
                'device': 'cuda:0',
                'trigger_word': trigger_word,
                'network': {
                    'type': 'lora',
                    'linear': lora_rank,
                    'linear_alpha': lora_rank,
                },
                'save': {
                    'dtype': 'float16',
                    'save_every': min(500, training_steps),
                    'max_step_saves_to_keep': 2,
                    'push_to_hub': False,
                },
                'datasets': [{
                    'folder_path': image_dir,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'shuffle_tokens': False,
                    'cache_latents_to_disk': True,
                    'resolution': [512, 768, resolution],
                }],
                'train': {
                    'batch_size': 1,
                    'steps': training_steps,
                    'gradient_accumulation_steps': 1,
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'timestep_type': 'weighted',
                    'optimizer': 'adamw8bit',
                    'lr': 1e-4,
                    'ema_config': {
                        'use_ema': True,
                        'ema_decay': 0.99,
                    },
                    'dtype': 'bf16',
                },
                'model': {
                    'name_or_path': model_id,
                    'arch': 'flux2',
                    'quantize': True,
                    'quantize_te': True,
                    'qtype': 'qfloat8',
                    'low_vram': True,
                    'model_kwargs': {
                        'match_target_res': False,
                    },
                },
                'sample': {
                    'sampler': 'flowmatch',
                    'sample_every': min(500, training_steps),
                    'width': 1024,
                    'height': 1024,
                    'prompts': [
                        f"photo of {trigger_word}, woman standing in sunlit apartment, natural light, warm tones",
                        f"photo of {trigger_word}, portrait, soft natural lighting, shallow depth of field",
                        f"photo of {trigger_word}, woman sitting casually, real interior, candid pose",
                    ],
                    'neg': '',
                    'seed': 42,
                    'walk_seed': True,
                    'guidance_scale': 4,
                    'sample_steps': 20,
                },
            }],
        },
    }
    return config


def fire_callback(callback_url, payload):
    if not callback_url:
        return
    try:
        import requests
        print(f"[TRAIN] Firing callback to {callback_url}", flush=True)
        resp = requests.post(callback_url, json=payload, timeout=30,
                             headers={'Content-Type': 'application/json'})
        print(f"[TRAIN] Callback: {resp.status_code} {resp.text[:200]}", flush=True)
    except Exception as e:
        print(f"[TRAIN] Callback failed: {e}", flush=True)


def train(zip_url, trigger_word='escort_person', training_steps=1500,
          lora_rank=32, resolution=1024, hf_token='', lora_id='',
          network_volume='/runpod-volume'):
    t_start = time.time()

    print("=" * 60, flush=True)
    print("[TRAIN] TGND Escort LoRA Training (ai-toolkit)", flush=True)
    print(f"[TRAIN] Trigger: {trigger_word}", flush=True)
    print(f"[TRAIN] Steps: {training_steps}, Rank: {lora_rank}, Res: {resolution}", flush=True)
    print(f"[TRAIN] LoRA ID: {lora_id}", flush=True)
    print("=" * 60, flush=True)

    if not zip_url:
        raise ValueError('No training ZIP URL provided')

    # ─── HuggingFace auth ───
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

    # ─── Download + extract training images ───
    print("[TRAIN] Downloading training images...", flush=True)
    image_dir = "/data/images"
    os.makedirs(image_dir, exist_ok=True)
    zip_path = "/data/training.zip"

    download_file(zip_url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(image_dir)

    # Flatten subdirectories
    for item in os.listdir(image_dir):
        sub = os.path.join(image_dir, item)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                src = os.path.join(sub, f)
                dst = os.path.join(image_dir, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(sub, ignore_errors=True)

    os.remove(zip_path)

    # ─── Validate images ───
    images = validate_images(image_dir)
    count = len(images)
    print(f"[TRAIN] Found {count} valid training images", flush=True)

    if count < 5:
        raise ValueError(f'Too few images: {count}')
    if count > 150:
        for img in sorted(images)[150:]:
            os.remove(img)

    # ─── Generate captions ───
    generate_captions(image_dir, trigger_word)

    # ─── Detect GPU ───
    gpu = detect_gpu()
    if gpu['vram_gb'] < 80:
        print(f"[TRAIN] WARNING: {gpu['vram_gb']:.0f}GB VRAM — FLUX.2 needs 80GB+ with quantization", flush=True)

    # ─── Build config ───
    model_id = "black-forest-labs/FLUX.2-dev"
    output_dir = "/output/escort-lora"
    os.makedirs(output_dir, exist_ok=True)

    config = build_training_config(
        trigger_word=trigger_word,
        image_dir=image_dir,
        output_dir=output_dir,
        model_id=model_id,
        training_steps=training_steps,
        lora_rank=lora_rank,
        resolution=resolution,
    )

    config_path = "/data/training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[TRAIN] Config written to {config_path}", flush=True)
    with open(config_path) as f:
        print(f.read(), flush=True)

    # ─── Run ai-toolkit training ───
    print("\n" + "=" * 60, flush=True)
    print("[TRAIN] Starting ai-toolkit LoRA training", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    run(f"cd /app/ai-toolkit && python run.py {config_path}", stream=True)

    train_elapsed = time.time() - t0
    print(f"[TRAIN] Training completed in {train_elapsed / 60:.1f} minutes", flush=True)

    # ─── Find output LoRA ───
    lora_file = None
    config_name = config['config']['name']
    lora_dir = os.path.join(output_dir, config_name)

    if os.path.isdir(lora_dir):
        candidates = []
        for f in os.listdir(lora_dir):
            if f.endswith('.safetensors'):
                fpath = os.path.join(lora_dir, f)
                candidates.append((os.path.getmtime(fpath), fpath))
        if candidates:
            candidates.sort(reverse=True)
            lora_file = candidates[0][1]

    if not lora_file:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith('.safetensors'):
                    fpath = os.path.join(root, f)
                    if lora_file is None or os.path.getmtime(fpath) > os.path.getmtime(lora_file):
                        lora_file = fpath

    if not lora_file or not os.path.exists(lora_file):
        print("[TRAIN] Output directory contents:", flush=True)
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                fpath = os.path.join(root, f)
                print(f"  {fpath} ({os.path.getsize(fpath)/1024/1024:.1f}MB)", flush=True)
        raise RuntimeError('LoRA weights file not produced')

    lora_size_mb = os.path.getsize(lora_file) / 1024 / 1024
    print(f"[TRAIN] LoRA weights: {lora_file} ({lora_size_mb:.1f}MB)", flush=True)

    # ─── Save: volume + HF Hub ───
    storage_key = ""
    dest_filename = f"escort_{lora_id}.safetensors"

    # Clean volume
    if os.path.exists(network_volume):
        for item in os.listdir(network_volume):
            if item == "loras":
                continue
            item_path = os.path.join(network_volume, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                elif os.path.isfile(item_path):
                    os.remove(item_path)
            except Exception:
                pass

    # Volume save
    volume_lora_dir = os.path.join(network_volume, "loras")
    if os.path.exists(network_volume):
        try:
            os.makedirs(volume_lora_dir, exist_ok=True)
            dest_path = os.path.join(volume_lora_dir, dest_filename)
            shutil.copy2(lora_file, dest_path)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) == os.path.getsize(lora_file):
                storage_key = dest_path
                print(f"[TRAIN] LoRA saved to volume: {storage_key}", flush=True)
        except OSError as e:
            print(f"[TRAIN] Volume save failed: {e}", flush=True)

    # HF Hub save
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            hf_repo = "JulioIglesiass/tgnd-loras"
            try:
                api.create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)
            except Exception:
                pass
            api.upload_file(path_or_fileobj=lora_file, path_in_repo=dest_filename,
                            repo_id=hf_repo, repo_type="model")
            hf_key = f"hf://{hf_repo}/{dest_filename}"
            print(f"[TRAIN] LoRA uploaded to HF: {hf_key}", flush=True)
            if not storage_key:
                storage_key = hf_key
        except Exception as e:
            print(f"[TRAIN] HF upload failed: {e}", flush=True)

    if not storage_key:
        print("[TRAIN] WARNING: LoRA not saved to persistent storage!", flush=True)

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

    print(f"\n[TRAIN] ALL DONE in {total_elapsed / 60:.1f} minutes", flush=True)
    print(f"[TRAIN] LoRA: {storage_key}", flush=True)
    return result


def main():
    zip_url = os.environ.get('TRAINING_ZIP_URL', '')
    trigger_word = os.environ.get('TRIGGER_WORD', 'escort_person')
    training_steps = int(os.environ.get('TRAINING_STEPS', '1500'))
    lora_rank = int(os.environ.get('LORA_RANK', '32'))
    resolution = int(os.environ.get('RESOLUTION', '1024'))
    hf_token = os.environ.get('HF_TOKEN', '')
    callback_url = os.environ.get('CALLBACK_URL', '')
    lora_id = os.environ.get('LORA_ID', '')
    network_volume = os.environ.get('NETWORK_VOLUME', '/runpod-volume')
    webhook_secret = os.environ.get('WEBHOOK_SECRET', '')

    try:
        result = train(zip_url=zip_url, trigger_word=trigger_word,
                       training_steps=training_steps, lora_rank=lora_rank,
                       resolution=resolution, hf_token=hf_token,
                       lora_id=lora_id, network_volume=network_volume)
        result['lora_id'] = lora_id
        result['secret'] = webhook_secret
        fire_callback(callback_url, result)
    except Exception as e:
        print(f"[TRAIN] FAILED: {e}", flush=True)
        fire_callback(callback_url, {'lora_id': lora_id, 'status': 'failed',
                                      'error': str(e), 'secret': webhook_secret})
        sys.exit(1)


if __name__ == "__main__":
    main()
