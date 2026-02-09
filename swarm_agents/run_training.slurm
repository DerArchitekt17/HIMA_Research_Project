#!/bin/bash
#SBATCH --job-name=hima_swarm_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:2          # Adjust: 2, 4, or 8 GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00                 # ~48h for 2 GPUs, ~12h for 8 GPUs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
TRAIN_SCRIPT=${BASE_DIR}/train.py

# Auto-detect available GPUs
NUM_GPUS=$(apptainer exec --nv --bind ${BASE_DIR}:${BASE_DIR} ${SIF} \
    python -c "import torch; print(torch.cuda.device_count())")
echo "Detected ${NUM_GPUS} GPU(s)"

# Configure online services for offline use
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Create directories
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/finetuned_models ${BASE_DIR}/wandb

# Train all 12 agents sequentially — each uses ALL available GPUs via accelerate
ROLES=("drafter" "critic" "refiner")
AGENTS=("subjective" "objective" "assessment" "plan")

CURRENT=0
TOTAL=$(( ${#ROLES[@]} * ${#AGENTS[@]} ))

for ROLE in "${ROLES[@]}"; do
    for AGENT in "${AGENTS[@]}"; do
        CURRENT=$((CURRENT + 1))

        export WANDB_PROJECT="hima_swarm_finetune"
        export WANDB_NAME="swarm_${ROLE}_${AGENT}"

        echo "========== [${CURRENT}/${TOTAL}] ${ROLE}/${AGENT} on ${NUM_GPUS} GPU(s) =========="

        apptainer exec --nv \
            --bind ${BASE_DIR}:${BASE_DIR} \
            ${SIF} \
            accelerate launch \
                --num_processes ${NUM_GPUS} \
                --mixed_precision no \
            ${TRAIN_SCRIPT} --role ${ROLE} --agent ${AGENT}

        echo "========== [${CURRENT}/${TOTAL}] ${ROLE}/${AGENT} done =========="
    done
done

echo "All ${TOTAL} swarm agents trained."
