import os
import json
from collections import defaultdict

from llmmark.clients.openai_client import OpenAIClient
from llmmark.evaluation.llm_evaluator import LLMJudge

JSON_PATH = "data/runs/run_1/gemma3:1b/question_1.json"

def load_answers(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def group_by_question(data):
    questions = defaultdict(list)
    for item in data:
        if "question" in item:
            questions[item["question"]].append(item)
    return questions

def eval_questions(judge, grouped_answers):
    results = []

    for question_text, runs in grouped_answers.items():
        first_run = runs[0]
        expected_answer = first_run.get("correct_answer", "")
        model_raw_answer = first_run.get("raw_answer", "")

        print("\n===============================")
        print(f"Question: {question_text.strip()}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Model Answer (raw): {model_raw_answer}")

        result = judge.eval(
            question=question_text.strip(),
            expected_answer=expected_answer.strip(),
            model_answer=model_raw_answer.strip()
        )

        print("\n--- Evaluation ---")
        print(f"Score: {result.get('grade')}")
        print(f"Justification: {result.get('justification')}")
        print(f"Latency: {result.get('latency')} seconds")
        results.append(result)

    return results

if __name__ == "__main__":
    data = load_answers(JSON_PATH)

    # Ignore aggregated statistics if present at the beginning of the data list
    # This assumes the first item might be a summary object
    if "num_correct" in data[0]:
        data = data[1:]

    questions_by_group = group_by_question(data)

    api_key = os.environ.get("GEMINI_API_KEY")
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = "gemini-2.0-flash"

    client = OpenAIClient(api_key=api_key, base_url=base_url)

    judge = LLMJudge(client=client)
    eval_questions(judge, questions_by_group)
