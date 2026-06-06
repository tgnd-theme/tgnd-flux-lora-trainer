#!/bin/bash
# Smart entrypoint: if RUNPOD_ENDPOINT_ID is set, we're in serverless mode.
# Otherwise, run standalone training script (pod mode).

if [ -n "$RUNPOD_ENDPOINT_ID" ]; then
    echo "[ENTRYPOINT] Serverless mode (endpoint: $RUNPOD_ENDPOINT_ID)"
    exec python3 -u /app/handler.py
else
    echo "[ENTRYPOINT] Pod mode — running training script"
    exec python3 -u /app/train_escort_lora.py
fi
