#!/bin/bash
# Setting environment variables
NUMBER_OF_SAMPLES=0   # set to 0 to run all samples
BASE_DIR="$(pwd)"
SCRIPT=${BASE_DIR}/benchmark.py
BENCHMARK_OUTPUT_FOLDER=${BASE_DIR}/benchmark_results

# Create directories
mkdir -p ${BENCHMARK_OUTPUT_FOLDER}

# Args processing
NUM_SAMPLES_ARG=""
if [ ${NUMBER_OF_SAMPLES} -ge 1 ]; then
    NUM_SAMPLES_ARG="--num_samples ${NUMBER_OF_SAMPLES}"
fi

# Run benchmark (auto-detects all available GPUs via torch.cuda.device_count)
python ${SCRIPT} ${NUM_SAMPLES_ARG} \
    --output ${BENCHMARK_OUTPUT_FOLDER}/base_data_benchmark_n${NUMBER_OF_SAMPLES}.json

echo "Benchmark complete."
