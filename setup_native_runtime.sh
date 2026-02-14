#!/bin/bash
apt update && apt upgrade -y

pip install --no-cache-dir \
        transformers==4.57.6 \
        datasets==3.2.0 \
        accelerate==1.7.0 \
        peft==0.14.0 \
        trl==0.13.0 \
        bitsandbytes \
        wandb \
        mistral-common \
        scipy \
        llmcompressor \
        rouge-score \
        bert-score \
        ninja

# Check for CUDA availability
python -m bitsandbytes

# Fix Triton mismatch [common CUDA error]
pip install -U --no-cache-dir "bitsandbytes>=0.45.3"

export MAX_JOBS=$(nproc)
export WANDB_MODE=offline

# Erst nachdem die Daten heruntergeladen wurden ausführen!
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

apt-get install -y build-essential ninja-build

pip install --no-cache-dir flash-attn --no-build-isolation