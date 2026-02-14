#!/bin/bash

# Require CUDA >= 13.0 (needed for flash-attn and GPU workloads)
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

if [ -z "$CUDA_VERSION" ]; then
    echo "ERROR: nvcc not found. CUDA toolkit is not installed or not in PATH."
    exit 1
elif [ "$CUDA_MAJOR" -lt 13 ]; then
    echo "ERROR: CUDA >= 13.0 required, found CUDA $CUDA_VERSION"
    exit 1
fi
echo "CUDA $CUDA_VERSION detected."

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

# Setup gcloud utilities
apt-get install apt-transport-https ca-certificates gnupg curl -y
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get install google-cloud-cli -y

# OBSZOLET: Erst nachdem die Daten heruntergeladen wurden ausführen!
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

apt-get install -y build-essential ninja-build

pip install --no-cache-dir flash-attn --no-build-isolation