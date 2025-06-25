#!/bin/bash

# ==============================================================================
# LLMmark - Determinism
# ==============================================================================

# Configuration

MODEL_ID="all"
NUM_RUNS=10
EXERCISE_FOLDER="exam_01_oa"
QUESTION_TYPE="open_answer"
LANGUAGE="en"
PROMPT_TECHNIQUES=(
    "R1"
    "R2"
    "R3"
    "R4"
)

TEMPERATURES=(
    0.0
    0.2
    0.4
)

echo "Starting LLMmark determinism test..."
echo "-------------------------------------"
echo -e "Model: \e[1;33m$MODEL_ID\e[0m"
echo -e "Repetitions per question: \e[1;33m$NUM_RUNS\e[0m"
echo "-------------------------------------"

cd "$(dirname "$0")/src" || exit

for TECHNIQUE in "${PROMPT_TECHNIQUES[@]}"; do

    for TEMP in "${TEMPERATURES[@]}"; do

        echo -e "\n\e[1;34m[RUNNING]\e[0m Technique: \e[1;33m$TECHNIQUE\e[0m | Temperature: \e[1;33m$TEMP\e[0m"
        echo "--------------------------------------------------------"

        python -m llmmark.determinism.main \
            --question-type "$QUESTION_TYPE" \
            --exercise-folder "$EXERCISE_FOLDER" \
            --language "$LANGUAGE" \
            --model-source "local" \
            --model-id "$MODEL_ID" \
            --prompting-technique "$TECHNIQUE" \
            --num-runs "$NUM_RUNS" \
            --temperature "$TEMP"

    done
done

echo -e "\n\e[1;32mDeterminism test completed for all techniques and temperatures.\e[0m"
echo "Check the output JSON files and the 'LLMmark_determinism' project on Opik/Comet to analyze response consistency."