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

# Create directories
mkdir -p ${SCRIPT_DIR}/finetuned_adapters ${SCRIPT_DIR}/wandb

# Train all 12 agents sequentially — each uses ALL available GPUs via accelerate
CURRENT=0
TOTAL=12

for ROLE in drafter critic refiner; do
    for AGENT in subjective objective assessment plan; do
        CURRENT=$((CURRENT + 1))

        export WANDB_PROJECT="hima_swarm_small_finetune"
        export WANDB_NAME="${ROLE}_${AGENT}"

        echo "========== [${CURRENT}/${TOTAL}] ${ROLE}/${AGENT} on ${NUM_GPUS} GPU(s) =========="
            accelerate launch \
                --num_processes ${NUM_GPUS} \
                --mixed_precision no \
            ${TRAIN_SCRIPT} --role ${ROLE} --agent ${AGENT}
        echo "========== [${CURRENT}/${TOTAL}] ${ROLE}/${AGENT} done =========="
    done
done

echo "All ${TOTAL} swarm agents trained."
