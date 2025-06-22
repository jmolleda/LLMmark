#!/bin/bash

# ==============================================================================
# LLMmark - Evaluation Runner Script
# ==============================================================================

# --- Configuration ---
# To evaluate all runs: RUN_FOLDERS_TO_EVALUATE=($(ls -d ../data/runs/run_* | xargs -n 1 basename))

RUN_FOLDERS_TO_EVALUATE=(
    "run_104"
    # "run_086"
    # "run_087"
)

# Evaluation

echo "Starting LLMmark evaluation process..."
echo "-------------------------------------"

cd "$(dirname "$0")/src" || exit

for RUN_FOLDER in "${RUN_FOLDERS_TO_EVALUATE[@]}"; do
    RUN_PATH="../data/runs/$RUN_FOLDER"

    if [ ! -d "$RUN_PATH" ]; then
        echo -e "\e[1;31m[ERROR]\e[0m Run folder '$RUN_FOLDER' not found. Skipping."
        continue
    fi

    echo -e "\n\e[1;34m[EVALUATING]\e[0m Run Folder: \e[1;33m$RUN_FOLDER\e[0m"

    python -m llmmark.evaluation.main --run-folder "$RUN_FOLDER"

    echo -e "\e[1;32m[COMPLETE]\e[0m Evaluation finished for $RUN_FOLDER."
    echo "-------------------------------------"

done

echo -e "\n\e[1;32mAll specified evaluations have been completed.\e[0m"

