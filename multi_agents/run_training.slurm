#!/bin/bash
#SBATCH --job-name=hima_multi_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
TRAIN_SCRIPT=${BASE_DIR}/train.py

# Configure online services for offline use
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# WANDB parameters
export WANDB_PROJECT="hima_multi_finetune"
export WANDB_NAME="hima_multi_finetune_run"

# Create directories
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/finetuned_models ${BASE_DIR}/wandb

# Batch 1 finetuning: subjective + objective
echo "========== Batch 1: subjective + objective =========="
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${TRAIN_SCRIPT} --agent subjective &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${TRAIN_SCRIPT} --agent objective &
wait
echo "========== Batch 1 done =========="

# Batch 2 finetuning: assessment + plan
echo "========== Batch 2: assessment + plan =========="
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${TRAIN_SCRIPT} --agent assessment &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${TRAIN_SCRIPT} --agent plan &
wait
echo "========== Batch 2 done =========="

echo "All agents trained."
