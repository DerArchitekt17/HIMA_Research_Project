#!/bin/bash
#SBATCH --job-name=hima_single_small_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_ROOT}/single_agent_small"
SIF="apptainer_runtime.sif"
TRAIN_SCRIPT=${SCRIPT_DIR}/train.py

# Auto-detect allocated GPUs (SLURM env -> CUDA_VISIBLE_DEVICES -> torch fallback)
if [ -n "${SLURM_GPUS_ON_NODE}" ]; then
    NUM_GPUS=${SLURM_GPUS_ON_NODE}
elif [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_GPUS=$(apptainer exec --nv --bind ${PROJECT_ROOT}:${PROJECT_ROOT} ${SIF} \
        python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
fi
echo "Allocated ${NUM_GPUS} GPU(s) for training"

# Create directories
mkdir -p ${SCRIPT_DIR}/logs ${SCRIPT_DIR}/finetuned_adapters ${SCRIPT_DIR}/wandb

export WANDB_PROJECT="hima_single_small_finetune"
export WANDB_NAME="single"

# Train model
echo "========== Training single SOAP model on ${NUM_GPUS} GPU(s) =========="
apptainer exec --nv \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    ${SIF} \
    accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision no \
    ${TRAIN_SCRIPT}
echo "========== Training done =========="

echo "Single model trained."
