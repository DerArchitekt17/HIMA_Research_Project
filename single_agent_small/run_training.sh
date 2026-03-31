#!/bin/bash
# Setting environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT=${SCRIPT_DIR}/train.py

# Auto-detect allocated GPUs
if [ -n "${GPUS_ON_NODE}" ]; then
    NUM_GPUS=${GPUS_ON_NODE}
elif [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
fi
echo "Allocated ${NUM_GPUS} GPU(s) for training"

# WANDB parameters
export WANDB_PROJECT="hima_single_small_finetune"
export WANDB_NAME="single_small_run"

# Create directories
mkdir -p ${SCRIPT_DIR}/finetuned_adapters ${SCRIPT_DIR}/wandb

# Train model
echo "========== Training single SOAP model =========="
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_GPUS} \
    ${TRAIN_SCRIPT}
echo "========== Training done =========="

echo "Single model trained."
