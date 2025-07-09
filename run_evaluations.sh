#!/bin/bash

# ==============================================================================
# LLMmark - Evaluation Runner Script
# ==============================================================================

# RUN_FOLDERS_TO_EVALUATE=(
#     "run_001"
#     "run_002"
# )

RUN_FOLDERS_TO_EVALUATE=($(find /home/cim/LLMmark/data/runs -maxdepth 1 -type d -regextype posix-extended -regex '.*/run_[0-9]{3,4}$' -printf "%f\n" 2>/dev/null))

echo "Run folders to evaluate:"
for RUN_FOLDER in "${RUN_FOLDERS_TO_EVALUATE[@]}"; do
    echo "- $RUN_FOLDER"
done

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

