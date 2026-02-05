#!/bin/bash
ENV_NAME="hima_research"
ENVIRONMENT_FILE="environment.yml"

conda create --name ${ENV_NAME}
conda env update --name ${ENV_NAME} --file ${ENVIRONMENT_FILE}
conda activate ${ENV_NAME}