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
PROMPT_TECHNIQUE="S1"
TEMPERATURE=0.0


echo "Starting LLMmark determinism test..."
echo "-------------------------------------"
echo -e "Model: \e[1;33m$MODEL_ID\e[0m"
echo -e "Repetitions per question: \e[1;33m$NUM_RUNS\e[0m"
echo "-------------------------------------"

cd "$(dirname "$0")/src" || exit

python -m llmmark.determinism.main \
    --question-type "$QUESTION_TYPE" \
    --exercise-folder "$EXERCISE_FOLDER" \
    --language "$LANGUAGE" \
    --model-source "local" \
    --model-id "$MODEL_ID" \
    --prompting-technique "$PROMPT_TECHNIQUE" \
    --num-runs "$NUM_RUNS" \
    --temperature "$TEMPERATURE"

echo -e "\n\e[1;32mDeterminism test completed.\e[0m"
echo "Check the output JSON files and the 'LLMmark_determinism' project on Opik/Comet to analyze response consistency."

