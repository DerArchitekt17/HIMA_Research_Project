#!/bin/bash
#SBATCH --job-name=hima_swarm_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_ROOT}/swarm_agents_small"
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

# Train all 12 agents sequentially — each uses ALL available GPUs via accelerate
CURRENT=0
TOTAL=12

for ROLE in drafter critic refiner; do
    for AGENT in subjective objective assessment plan; do
        CURRENT=$((CURRENT + 1))

        export WANDB_PROJECT="hima_swarm_small_finetune"
        export WANDB_NAME="${ROLE}_${AGENT}"

        echo "========== [${CURRENT}/${TOTAL}] ${ROLE}/${AGENT} on ${NUM_GPUS} GPU(s) =========="

        apptainer exec --nv \
            --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
            ${SIF} \
            accelerate launch \
                --num_processes ${NUM_GPUS} \
                --mixed_precision no \
            ${TRAIN_SCRIPT} --role ${ROLE} --agent ${AGENT}

        echo "========== [${CURRENT}/${TOTAL}] ${ROLE}/${AGENT} done =========="
    done
done

echo "All ${TOTAL} swarm agents trained."
