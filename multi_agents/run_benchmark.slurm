#!/bin/bash
#SBATCH --job-name=hima_multi_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
NUMBER_OF_SAMPLES=0   # set to 0 to run all samples. 2,006 samples available
BATCH_SIZE=8          # samples per GPU per batch (reduce if OOM; multi-agent prompts grow with soap_so_far)
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_ROOT}/multi_agents"
SIF="apptainer_runtime.sif"
SCRIPT=${SCRIPT_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER=${SCRIPT_DIR}/benchmark_results

# Create directories
mkdir -p ${SCRIPT_DIR}/logs ${BENCHMARK_OUTPUT_FOLDER}

# Args processing
NUM_SAMPLES_ARG=""
if [ ${NUMBER_OF_SAMPLES} -ge 1 ]; then
    NUM_SAMPLES_ARG="--num_samples ${NUMBER_OF_SAMPLES}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run benchmark
apptainer exec --nv \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    ${SIF} \
    python ${SCRIPT} ${NUM_SAMPLES_ARG} \
    --batch_size ${BATCH_SIZE} \
    --output ${BENCHMARK_OUTPUT_FOLDER}/hima_multi_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
