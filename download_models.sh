#!/bin/bash
# Download models for offline use. Run ONCE with internet access.
# Models are stored in a shared models/ directory at the project root.
MODEL_NAME="mistralai/Ministral-3-8B-Reasoning-2512"
MODEL_NAME_SMALL="mistralai/Ministral-3-3B-Reasoning-2512"
BERTSCORE_MODEL="roberta-large"

# Download base LLM (8B)
echo "Downloading ${MODEL_NAME} ..."
hf download "${MODEL_NAME}" --local-dir "models/ministral3_8B"

# Download base LLM (3B)
echo "Downloading ${MODEL_NAME_SMALL} ..."
hf download "${MODEL_NAME_SMALL}" --local-dir "models/ministral3_3B"

# Download model for BERTScore
echo "Downloading ${BERTSCORE_MODEL} ..."
hf download "${BERTSCORE_MODEL}" --local-dir "models/bertscore"

echo "Done."
