import os
import re
import json
import requests
from ollama_client import OllamaClient
from settings import Settings
import time


def open_dataset_folder(settings):
    """Open the folder containing the question dataset."""
    folder = settings.folders['data_folder_name']
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        exit(1)
    print(f"Question dataset folder: \033[92m{folder}\033[0m")
    return folder

def get_models(models):
    """Return a list of (display_name, model_id) tuples"""
    return [(model['display_name'], model['model_id']) for model in models]

def get_questions_from_folder(folder, settings):
    """Read all question files from the given folder"""
    questions = []
    prefix = settings.files['question_file_name']
    ext = settings.files['question_file_extension']
    for filename in sorted(os.listdir(folder)):
        if filename.startswith(prefix) and filename.endswith(ext):
            file_path = os.path.join(folder, filename)
            print(f"Reading file: {file_path}")
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                continue
            with open(file_path, "r") as f:
                questions.append((filename, f.read().strip()))
    return questions

def create_run_folder(settings):
    """Create a new run folder for experiment outputs."""
    base_folder = settings.folders['base_experiments_folder']
    os.makedirs(base_folder, exist_ok=True)
    prefix = settings.folders['experiment_folder_name']
    existing = [d for d in os.listdir(base_folder) if d.startswith(prefix) and os.path.isdir(os.path.join(base_folder, d))]
    run_numbers = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    n = max(run_numbers, default=0) + 1
    run_folder = os.path.join(base_folder, f"{prefix}{n}")
    os.makedirs(run_folder)
    print(f"Created run experiment folder: {run_folder}")
    return run_folder

def get_llm_response(settings, client, model, prompt):
    if settings.model_mode == 'chat':
        response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    if settings.model_mode == 'generate':
        response = client.generate(model=model, prompt=prompt)
        return response['response']
    return None
    
    
def is_model_installed(model_name: str, ollama_base_url) -> bool:
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(model.get("name", "").startswith(model_name) for model in models)
    except requests.exceptions.RequestException as e:
        print(f"Error checking model list: {e}")
        return False    
    
def pull_model(model_name: str, ollama_base_url) -> None:
    try:
        response = requests.post(
            f"{ollama_base_url}/api/pull",
            json={"name": model_name},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    except requests.exceptions.RequestException as e:
        print(f"Error pulling model: {e}")    

def select_model(models, host):
    """Prompt the user to select a model from config, or all models. If not installed, offer to pull it."""
    print("Available Ollama models:")
    for idx, (name, _) in enumerate(models, 1):
        print(f"{idx}. {name}")

    choice = input("Enter model number or 'A' to run all models (default [1]): ").strip().lower()
    if choice == 'a':
        return [m[1] for m in models], True

    if not choice or not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        selected_model = models[0][1]
    else:
        selected_model = models[int(choice) - 1][1]
    print(f"Selected model: \033[92m{selected_model}\033[0m")

    if not is_model_installed(selected_model, host):
        print(f"{selected_model} is not available in Ollama.")
        confirm = input(f"Do you want to pull '{selected_model}'? (Y/n): ").strip().lower()
        if confirm == '' or confirm == 'y':
            pull_model(selected_model, host)
            print(f"Pulling {selected_model}...")
            if is_model_installed(selected_model, host):
                print(f"{selected_model} successfully installed.")
            else:
                print(f"Failed to install {selected_model}.")
        else:
            print("Installation cancelled.")      

    return selected_model, False

def select_question_type(settings):
    """Select open-answer or multiple-choice questions and return the base prompt."""
    print("Question type selected: ", end="")
    if settings.question_type == 'multiple_choice':
        print("\033[92mmultiple choice\033[0m")
        return settings.prompts['multiple_choice_questions']
    
    if settings.question_type == 'open_answer':
        print("\033[92mopen answer\033[0m")
        return settings.prompts['open_answer_questions']
    
    return None

def run_questions_for_model(model_display_name, model_id, questions, settings, client, run_folder, base_prompt):
    print(f"\n\033[92m=== Running questions for model: {model_display_name} ({model_id}) ===\033[0m")
    model_run_folder = os.path.join(run_folder, model_id)
    os.makedirs(model_run_folder, exist_ok=True)

    for idx, (filename, question) in enumerate(questions, 1):
        #Get the correct answer from the question, and remove it from the question text
        match = re.search(r'<(.*?)>', question)
        correct_answer = "[" + match.group(1)  + "]" if match else ''
        question = re.sub(r'<.*?>', '', question)
        question = question[:-1]

        prompt = base_prompt + question
        print(f"\n{prompt}")
        outputs = []
        correct_answers = 0
        total_response_time = 0
        for run_idx in range(settings.num_runs_per_question):
            # Display experiment name, model name, question number, and iteration
            exp_name = os.path.basename(run_folder) if run_folder else 'N/A'
            q_str = f"{idx:2}"  # 2-character width for question number
            iter_str = f"{run_idx + 1:2}"  # 2-character width for iteration
            print(f"Experiment: {exp_name} | Model: {model_display_name} | Question: {q_str} | Iteration: {iter_str} | Answer: ", end="")

            start_time = time.time()
            answer = get_llm_response(settings, client, model_id, prompt)
            response_time = round(time.time() - start_time, 3)
            total_response_time += response_time
            
            answer_no_newlines = answer.replace('\n', '') if answer else ''
            answer_no_think = re.sub(r'<think>.*?</think>', '', answer_no_newlines, flags=re.DOTALL).strip() if answer_no_newlines else ''
            # Keep only text matching the pattern [*], where * is a single character
            matches = re.findall(r'\[[^\]]\]', answer_no_think)
            answer_clean = ''.join(matches)                
            
            if answer_clean == correct_answer:
                print(f"\033[92m{answer_clean}\033[0m")  # Green
                correct_answers += 1
            else:
                print(f"\033[91m{answer_clean}\033[0m")  # Red

            outputs.append({
                "question": question,
                "answer": answer_clean,
                "correct_answer": correct_answer,
                "raw_answer": answer_no_think,
                "model": model_display_name,
                "response_time (s.)": response_time,
            })

        # Calculate statistics
        accuracy = round(correct_answers / settings.num_runs_per_question, 2)
        avg_response_time = round(total_response_time / settings.num_runs_per_question, 3)
        # Insert statistics at the beginning of the outputs list
        outputs.insert(0, {
            "Num. correct answers": correct_answers,
            "Accuracy": accuracy,
            "Avg_response_time (s.)": avg_response_time,
        })

        out_file = os.path.join(model_run_folder, f"{settings.files['question_file_name']}{idx}.json")
        print(f"Writing to file: {out_file}")
        with open(out_file, "w") as f:
            json.dump(outputs, f, indent=2)

def main():
    settings = Settings()

    # Initialize Ollama client
    client = OllamaClient(settings.ollama['host'])
    
    # Access question dataset folder
    folder = open_dataset_folder(settings)

    # Select model(s) to run
    models = get_models(settings.models)
    selected, run_all_models = select_model(models, settings.ollama['host'])    
    if run_all_models:
        selected_models = [(name, mid) for name, mid in models if mid in set(selected)]
    else:
        selected_models = [(next((name for name, mid in models if mid == selected), selected), selected)]    

    # Select question type
    base_prompt = select_question_type(settings)

    # Create run folder
    run_folder = create_run_folder(settings)

    # Read questions from the dataset folder
    questions = get_questions_from_folder(folder, settings)

    # Run questions for each selected model
    for model_display_name, model_id in selected_models:
        run_questions_for_model(model_display_name, model_id, questions, settings, client, run_folder, base_prompt)

if __name__ == "__main__":
    main()