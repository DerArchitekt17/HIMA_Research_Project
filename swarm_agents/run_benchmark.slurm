#!/bin/bash
#SBATCH --job-name=hima_swarm_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:8          # Adjust to available GPU count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Setting environment variables
NUMBER_OF_SAMPLES=500   # set to 0 to run all samples
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF="../apptainer_runtime.sif"
SCRIPT=${BASE_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER=${BASE_DIR}/benchmark_results

# Create directories
mkdir -p ${BASE_DIR}/logs ${BENCHMARK_OUTPUT_FOLDER}

# Configure online services for offline use
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Args processing
NUM_SAMPLES_ARG=""
if [ ${NUMBER_OF_SAMPLES} -ge 1 ]; then
    NUM_SAMPLES_ARG="--num_samples ${NUMBER_OF_SAMPLES}"
fi

# Run benchmark (auto-detects all available GPUs via torch.cuda.device_count)
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${SCRIPT} ${NUM_SAMPLES_ARG} \
    --output ${BENCHMARK_OUTPUT_FOLDER}/hima_swarm_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
