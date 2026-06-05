"""
RunPod serverless handler for TGND LoRA training.

Receives training parameters as job input (not env vars), runs DreamBooth
training via train_escort_lora.train(), fires the callback webhook
(belt-and-suspenders alongside RunPod's own webhook delivery), and returns
the result dict.
"""

import traceback
import runpod
from train_escort_lora import train, fire_callback


def handler(job):
    """RunPod serverless handler entry point."""
    inp = job.get("input", {})

    # Required
    zip_url = inp.get("zip_url", "")
    trigger_word = inp.get("trigger_word", "escort_person")
    lora_id = str(inp.get("lora_id", ""))

    # Optional with defaults
    training_steps = int(inp.get("training_steps", 1000))
    lora_rank = int(inp.get("lora_rank", 16))
    resolution = int(inp.get("resolution", 512))
    hf_token = inp.get("hf_token", "")
    network_volume = inp.get("network_volume", "/runpod-volume")
    callback_url = inp.get("callback_url", "")
    webhook_secret = inp.get("webhook_secret", "")

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

        # Belt-and-suspenders: fire our own callback in addition to RunPod webhook
        result["lora_id"] = lora_id
        result["secret"] = webhook_secret
        fire_callback(callback_url, result)

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"[HANDLER] Training failed: {error_msg}", flush=True)
        traceback.print_exc()

        # Fire failure callback
        fire_callback(callback_url, {
            "lora_id": lora_id,
            "status": "failed",
            "error": error_msg,
            "secret": webhook_secret,
        })

        raise  # Re-raise so RunPod marks job as FAILED


runpod.serverless.start({"handler": handler})
