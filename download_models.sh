#!/bin/bash
# Download models for offline use. Run ONCE with internet access.

MODEL_NAME="mistralai/Ministral-3-8B-Reasoning-2512"
OUTPUT_DIR="basemodel"

BERTSCORE_MODEL="roberta-large"
OUTPUT_BERTMODEL_DIR="bertscore_model"

mkdir -p "single_agent/${OUTPUT_DIR}" "single_agent/${OUTPUT_BERTMODEL_DIR}"
mkdir -p "multi_agents/${OUTPUT_DIR}" "multi_agents/${OUTPUT_BERTMODEL_DIR}"
mkdir -p "swarm_agents/${OUTPUT_DIR}" "swarm_agents/${OUTPUT_BERTMODEL_DIR}"

echo "Downloading ${MODEL_NAME} to ${OUTPUT_DIR} ..."
hf download "${MODEL_NAME}" --local-dir "single_agent/${OUTPUT_DIR}"
cp -r single_agent/${OUTPUT_DIR} multi_agents/${OUTPUT_DIR}
cp -r single_agent/${OUTPUT_DIR} swarm_agents/${OUTPUT_DIR}

echo "Downloading ${BERTSCORE_MODEL} to ${OUTPUT_BERTMODEL_DIR} ..."
hf download "${BERTSCORE_MODEL}" --local-dir "single_agent/${OUTPUT_BERTMODEL_DIR}"
cp -r single_agent/${OUTPUT_BERTMODEL_DIR} multi_agents/${OUTPUT_BERTMODEL_DIR}
cp -r single_agent/${OUTPUT_BERTMODEL_DIR} swarm_agents/${OUTPUT_BERTMODEL_DIR}

echo "Done."
