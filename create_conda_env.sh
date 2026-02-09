#!/bin/bash
ENV_NAME="hima_research"
ENVIRONMENT_FILE="environment.yml"

conda create --name ${ENV_NAME} -y
conda env update --name ${ENV_NAME} --file ${ENVIRONMENT_FILE}
conda init
conda activate ${ENV_NAME}