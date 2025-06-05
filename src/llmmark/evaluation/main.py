import os
import json
from collections import defaultdict
from ..settings import Settings
from .llm_evaluator import LLMJudge
from ..clients.openai_client import OpenAIClient

def load_model_answers(run_folder, model_id, question_file_prefix):
    model_path = os.path.join(run_folder, model_id)
    all_items = []
    for fname in sorted(os.listdir(model_path)):
        if fname.startswith(question_file_prefix) and fname.endswith(".json"):
            with open(os.path.join(model_path, fname), "r") as f:
                data = json.load(f)
                all_items.append((fname, data[1:]))  # omit stats
    return all_items

def group_by_question(data):
    grouped = defaultdict(list)
    for _, items in data:
        for entry in items:
            grouped[entry["question"]].append(entry)
    return grouped


def select_run(settings):
    base_folder = settings.experiments_folder

    print (f"Available runs in \033[1m{base_folder}\033[0m:")

    available_runs = sorted([
        d for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ])

    if not available_runs:
        print("No runs found.")
        return

    print("Available runs:")
    for idx, run in enumerate(available_runs, 1):
        print(f"{idx}. {run}")
    
    run_choice = input("Enter a run number (default [1]): ").strip()
    run_idx = int(run_choice) - 1 if run_choice.isdigit() and 1 <= int(run_choice) <= len(available_runs) else 0
    selected_run = available_runs[run_idx]
    run_path = os.path.join(base_folder, selected_run)

    return selected_run, run_path

def select_model(selected_run, run_path):
    available_models = sorted([
        d for d in os.listdir(run_path)
        if os.path.isdir(os.path.join(run_path, d))
    ])

    if not available_models:
        print("No models found in this run.")
        return

    print(f"\nAvailable models in \033[1m{selected_run}\033[0m:")
    for idx, model in enumerate(available_models, 1):
        print(f"{idx}. {model}")
    
    model_choice = input("Enter model number or 'A' to run all models (default [1]): ").strip()
    model_idx = int(model_choice) - 1 if model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models) else 0
    model_id = available_models[model_idx]
    model_path = os.path.join(run_path, model_id)

    print(f"\nSelected model: \033[92m{model_id}\033[0m")
    return model_id, model_path

def get_online_models(models):
    """Return a list of (display_name, model_id, base_url, api_key) tuples"""
    return [(model['display_name'], model['model_id'], model['base_url'], model['api_key'], model['max_requests_per_minute']) for model in models]


def main():
    settings = Settings()

    # STEP 1: Show all available runs
    selected_run, run_path = select_run(settings)

    # STEP 2: Show all models in selected run
    model_id, model_path = select_model(selected_run, run_path)

    # STEP 3: Load and group answers
    raw_data = load_model_answers(run_path, model_id, settings.question_file_prefix)
    grouped = group_by_question(raw_data)

    # STEP 4: Setup evaluator
    api_key = os.environ.get(settings.api_key_env_var)
    if not api_key:
        raise EnvironmentError(f"API key environment variable '{settings.api_key_env_var}' not set.")
    
    # with open("./settings/config.yaml", "r") as file:
    #     data = yaml.safe_load(file)
    #     model = next((m for m in data["openai_models"] if m["model_id"] == settings.model_name), None)
    #     max_requests_per_minute = model['max_requests_per_minute']
    #     print(f"\nSelected model: \033[92m{model['display_name']}\033[0m")
    #     print(f"Max requests per minute: {max_requests_per_minute}")

    models = get_online_models(settings.openai_models)
    selected_model = next((m for m in models if m[1] == settings.model_name), None)


    
    # base_url = models[selected_model][2]
    # api_key = os.environ.get(models[selected_model][3])
    # max_requests_per_minute = models[selected_model][4]


    if selected_model:
        display_name, model_id, base_url, api_key, max_requests_per_minute = selected_model

        api_key = os.environ.get(api_key, None)
        print(f"\n=== LLM \033[92m{display_name}\033[0m as a judge ===")
    else:
        print(f"Model with ID '{settings.model_name}' not found.")

        
    client = OpenAIClient(api_key=api_key, base_url=base_url, max_requests_per_minute=max_requests_per_minute)
    judge = LLMJudge(client=client, model=settings.model_name)

    # STEP 5: Evaluate each question
    results = []
    for q_idx, (question, answers) in enumerate(grouped.items(), 1):
        print(f"\n{q_idx:02d}. \033[1mQuestion:\033[0m {question.strip()}")
        expected = answers[0]["correct_answer"]

        for run_idx, run in enumerate(answers[:settings.num_runs_per_question], 1):
            model_output = run["raw_answer"]

            result = judge.eval(
                question=question.strip(),
                expected_answer=expected.strip(),
                model_answer=model_output.strip()
            )

            print(f"   Run {run_idx}:")
            print(f"     \033[93mModel Answer:\033[0m {model_output.strip()}")
            print(f"     \033[92mScore:\033[0m {result['grade']}")
            print(f"     \033[90mJustification:\033[0m {result['justification']}")
            print(f"     ⏱️  Latency: {result['latency']}s")

            results.append({
                "question": question,
                "run": run_idx,
                "grade": result["grade"],
                "justification": result["justification"],
                "latency": result["latency"]
            })


    # STEP 6: Save results
    output_path = os.path.join(model_path, settings.evaluation_output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()
