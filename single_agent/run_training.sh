#!/bin/bash
# Setting environment variables
BASE_DIR="$(pwd)"
TRAIN_SCRIPT=${BASE_DIR}/train.py

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
export WANDB_PROJECT="hima_single_finetune"
export WANDB_NAME="single_agent"

# Create directories
mkdir -p ${BASE_DIR}/finetuned_models ${BASE_DIR}/wandb

# Train model
echo "========== Training single SOAP model =========="
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_GPUS} \
    ${TRAIN_SCRIPT}
echo "========== Training done =========="

echo "Single model trained."
