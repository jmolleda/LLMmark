#!/bin/bash

# ==============================================================================
# LLMmark - Experiment Runner Script
# ==============================================================================

# Configuration
MODELS=(
    "gemma3n:e2b"
)


PROMPT_TECHNIQUES=(
    "S1"
    "S2"
    "S3"
    "S4"
    "R1"
    "R2"
    "R3"
    "R4"
    "D1"
    "D2"
    "D3"
    "D4"
)

EXERCISE_FOLDERS=(
    "exercises_03_mc"
    "exercises_04_mc"
    "exercises_05_mc"
)

QUESTION_TYPE="multiple_choice"
LANGUAGE="en"
NUM_RUNS=1
TEMPERATURE=0.0
TOP_P=0.1
MODEL_SOURCE="local"


echo "Starting LLMmark experiment grid..."
echo "-------------------------------------"


cd "$(dirname "$0")/src" || exit

for FOLDER in "${EXERCISE_FOLDERS[@]}"; do

    for TECHNIQUE in "${PROMPT_TECHNIQUES[@]}"; do

        for MODEL in "${MODELS[@]}"; do

            echo -e "\n\e[1;34m[RUNNING]\e[0m Model: \e[1;33m$MODEL\e[0m | Technique: \e[1;33m$TECHNIQUE\e[0m | Folder: \e[1;33m$FOLDER\e[0m"

            python -m llmmark.main \
                --question-type "$QUESTION_TYPE" \
                --exercise-folder "$FOLDER" \
                --language "$LANGUAGE" \
                --model-source "$MODEL_SOURCE" \
                --model-id "$MODEL" \
                --prompting-technique "$TECHNIQUE" \
                --num-runs "$NUM_RUNS" \
                --temperature "$TEMPERATURE" \
                --top-p "$TOP_P"

            echo -e "\e[1;32m[COMPLETE]\e[0m Finished run for Model: $MODEL, Technique: $TECHNIQUE"
            echo "-------------------------------------"
            sleep 2
        done
    done
done

echo -e "\n\e[1;32mAll experiments have been completed.\e[0m"

