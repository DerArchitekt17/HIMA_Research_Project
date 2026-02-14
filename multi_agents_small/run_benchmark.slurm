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
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
SCRIPT=${BASE_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER=${BASE_DIR}/benchmark_results

# Create directories
mkdir -p ${BASE_DIR}/logs ${BENCHMARK_OUTPUT_FOLDER}

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
    --output ${BENCHMARK_OUTPUT_FOLDER}/hima_multi_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
