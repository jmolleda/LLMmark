#!/bin/bash

# ==============================================================================
# LLMmark - Experiment Runner Script
# ==============================================================================

# Configuration
MODELS=(
    # "deepseek-r1:1.5b"
    "gemma3:1b"
    "gemma3:4b"
    "llama3.2:1b"
    # "moondream:1.8b"
    # "smollm2:1.7b"
    # "tinyllama:1.1b"
    # "qwen3:0.6b"
    # "qwen3:1.7b"
    # "qwen3:4b"
)


PROMPT_TECHNIQUES=(
    "R1"
    "S1"
    "D1"
)

EXERCISE_FOLDERS=(
    "exam_01_oa"
    "exercises_01_oa"
)

QUESTION_TYPE="open_answer"
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

