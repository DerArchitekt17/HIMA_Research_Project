#!/bin/bash
#SBATCH --job-name=hima_single_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
NUMBER_OF_SAMPLES=500   # set to 0 to run all samples
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
SCRIPT=${BASE_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER="/benchmark_results"

# Create directories
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/${BENCHMARK_OUTPUT_FOLDER}

# Args processing
NUM_SAMPLES_ARG=""
if [ ${NUMBER_OF_SAMPLES} -ge 1 ]; then
    NUM_SAMPLES_ARG="--num_samples ${NUMBER_OF_SAMPLES}"
fi

# Run benchmark
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${SCRIPT} ${NUM_SAMPLES_ARG} \
    --output ${BASE_DIR}/${BENCHMARK_OUTPUT_FOLDER}/hima_single_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
