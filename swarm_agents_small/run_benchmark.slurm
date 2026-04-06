#!/bin/bash
#SBATCH --job-name=hima_swarm_small_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
NUMBER_OF_SAMPLES=0   # set to 0 to run all samples. 2,006 samples available
BATCH_SIZE=32         # samples per GPU per batch (reduce if OOM)
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_ROOT}/swarm_agents_small"
SIF="apptainer_runtime.sif"
SCRIPT=${SCRIPT_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER=${SCRIPT_DIR}/benchmark_results

# Create directories
mkdir -p ${SCRIPT_DIR}/logs ${BENCHMARK_OUTPUT_FOLDER}

# Configure online services for offline use
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Args processing
NUM_SAMPLES_ARG=""
if [ ${NUMBER_OF_SAMPLES} -ge 1 ]; then
    NUM_SAMPLES_ARG="--num_samples ${NUMBER_OF_SAMPLES}"
fi

# Run benchmark
apptainer exec --nv \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    ${SIF} \
    python ${SCRIPT} ${NUM_SAMPLES_ARG} \
    --batch_size ${BATCH_SIZE} \
    --output ${BENCHMARK_OUTPUT_FOLDER}/hima_swarm_small_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
