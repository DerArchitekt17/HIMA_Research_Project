#!/bin/bash
#SBATCH --job-name=hima_single_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
TRAIN_SCRIPT=${BASE_DIR}/train.py

# Auto-detect allocated GPUs (SLURM env -> CUDA_VISIBLE_DEVICES -> torch fallback)
if [ -n "${SLURM_GPUS_ON_NODE}" ]; then
    NUM_GPUS=${SLURM_GPUS_ON_NODE}
elif [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_GPUS=$(apptainer exec --nv --bind ${BASE_DIR}:${BASE_DIR} ${SIF} \
        python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
fi
echo "Allocated ${NUM_GPUS} GPU(s) for training"

# Create directories
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/finetuned_models ${BASE_DIR}/wandb

export WANDB_PROJECT="hima_single_finetune"
export WANDB_NAME="hima_single_finetune_run"

# Train model
echo "========== Training single SOAP model on ${NUM_GPUS} GPU(s) =========="
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision no \
    ${TRAIN_SCRIPT}
echo "========== Training done =========="

echo "Single model trained."
