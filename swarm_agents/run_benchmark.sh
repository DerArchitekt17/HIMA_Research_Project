#!/bin/bash
# Setting environment variables
NUMBER_OF_SAMPLES=0   # set to 0 to run all samples. 2,006 samples available
BATCH_SIZE=16         # samples per GPU per batch (reduce if OOM)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT=${SCRIPT_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER=${SCRIPT_DIR}/benchmark_results

# Create directories
mkdir -p ${BENCHMARK_OUTPUT_FOLDER}

# Args processing
NUM_SAMPLES_ARG=""
if [ ${NUMBER_OF_SAMPLES} -ge 1 ]; then
    NUM_SAMPLES_ARG="--num_samples ${NUMBER_OF_SAMPLES}"
fi

# Run benchmark (auto-detects all available GPUs via torch.cuda.device_count)
python ${SCRIPT} ${NUM_SAMPLES_ARG} \
    --batch_size ${BATCH_SIZE} \
    --output ${BENCHMARK_OUTPUT_FOLDER}/hima_swarm_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
