#!/bin/bash
#SBATCH --job-name=hima_single_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
TRAIN_SCRIPT=${BASE_DIR}/train.py

# Configure online services for offline use
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="offline"

# WANDB parameters
export WANDB_PROJECT="hima_single_finetune"
export WANDB_NAME="hima_single_finetune_run"

# Create directories
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/finetuned_models ${BASE_DIR}/wandb

# Train model
echo "========== Training single SOAP model =========="
apptainer exec --nv \
  --bind ${BASE_DIR}:${BASE_DIR} \
  ${SIF} \
  accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    ${TRAIN_SCRIPT}
echo "========== Training done =========="

echo "Single model trained."
