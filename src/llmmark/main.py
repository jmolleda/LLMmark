import os
import re
import json
import time
import requests
from .settings import Settings
from .statistics import Statistics

from .clients.openai_client import OpenAIClient
from .clients.ollama_client import OllamaClient
from opik import Opik, LLMProvider
opik_client = Opik(project_name="LLMmark_response_generation")

def open_dataset_folder(settings):
    base_folder = settings.folders['data_folder_name']
    
    if not os.path.exists(base_folder) or not os.path.isdir(base_folder):
        print(f"Base data folder '{base_folder}' does not exist.")
        exit(1)

    try:
        # List all the exercise folders
        exercise_folders = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
    except OSError as e:
        print(f"Error reading subfolders from '{base_folder}': {e}")
        exit(1)

    if not exercise_folders:
        print(f"No exercise folders found in '{base_folder}'.")
        exit(1)

    print("\nPlease select an exercise folder:")
    for idx, folder_name in enumerate(exercise_folders, 1):
        print(f"  {idx}. {folder_name}")

    choice = input("Enter the number of the folder (default [1]): ").strip()
    
    selected_index = 0  # Default to the first folder
    if choice.isdigit() and 1 <= int(choice) <= len(exercise_folders):
        selected_index = int(choice) - 1
    elif choice:
        print(f"Invalid choice. Using default folder: {exercise_folders[0]}")

    selected_folder_name = exercise_folders[selected_index]
    selected_folder_path = os.path.join(base_folder, selected_folder_name)
    print(f"Selected folder: \033[92m{selected_folder_name}\033[0m")

    # Selection of language
    potential_languages = {"en": "English", "es": "Spanish"}
    found_languages = [lang for lang in potential_languages if os.path.isdir(os.path.join(selected_folder_path, lang))]

    # If no language folders are found, return the path to the selected exercise folder
    if not found_languages:
        return selected_folder_path

    # If language folders are found, prompt for selection
    print("\nLanguage options found. Please select one:")
    for idx, lang_code in enumerate(found_languages, 1):
        print(f"  {idx}. {potential_languages[lang_code]}")

    lang_choice = input("Enter language number (default [1] for English): ").strip()

    # Determine default language (prefer English if available)
    default_lang_idx = 0
    if "en" in found_languages:
        default_lang_idx = found_languages.index("en")

    selected_lang_idx = default_lang_idx
    if lang_choice.isdigit() and 1 <= int(lang_choice) <= len(found_languages):
        selected_lang_idx = int(lang_choice) - 1
    elif lang_choice:
        print("Invalid language choice. Using default.")

    final_lang_code = found_languages[selected_lang_idx]
    final_path = os.path.join(selected_folder_path, final_lang_code)
    
    print(f"Selected language: \033[92m{potential_languages[final_lang_code]}\033[0m")
    
    return final_path


def get_local_models(models):
    """Return a list of (display_name, model_id) tuples"""
    return [(model['display_name'], model['model_id']) for model in models]

def get_online_models(models):
    """Return a list of (display_name, model_id, base_url, api_key) tuples"""
    return [(model['display_name'], model['model_id'], model['base_url'], model['api_key'], model['max_requests_per_minute']) for model in models]

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
    run_folder_name = f"{prefix}{n:03d}"
    run_folder = os.path.join(base_folder, run_folder_name)
    os.makedirs(run_folder)
    print(f"Created run experiment folder: {run_folder}")
    return run_folder

def get_llm_response(settings, client, model, prompt):
    if settings.model_mode == 'chat':
        return client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    if settings.model_mode == 'generate':
        return client.generate(model=model, prompt=prompt)
    return ""
    
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

def select_local_model(models, host):
    client = OllamaClient(host)

    """Prompt the user to select a model from config, or all models. If not installed, offer to pull it."""
    print("Available Ollama models:")
    for idx, (name, _) in enumerate(models, 1):
        print(f"{idx}. {name}")

    choice = input("Enter model number or 'A' to run all models (default [1]): ").strip().lower()
    if choice == 'a':
        return [m[1] for m in models], True, client

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

    return selected_model, False, client

def select_online_model(models):
    """Prompt the user to select an online model or all models."""
    print("Available online models:")
    for idx, (name, _, _, _, _) in enumerate(models, 1):
        print(f"{idx}. {name}")

    choice = input("Enter model number (default [1]): ").strip().lower()

    if not choice or not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        idx = 0
    else:
        idx = int(choice) - 1

    selected_model = models[idx][1]
    base_url = models[idx][2]
    api_key = os.environ.get(models[idx][3])
    max_requests_per_minute = models[idx][4]

    print(f"Selected model: \033[92m{selected_model}\033[0m")
    print(f"Using API key: \033[92m{api_key}\033[0m")
    print(f"Using base URL: \033[92m{base_url}\033[0m")
    print(f"Max. requests per minute: \033[92m{max_requests_per_minute}\033[0m")

    client = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        max_requests_per_minute=max_requests_per_minute
    )

    return selected_model, False, client

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

def run_questions_for_model(model_display_name, model_id, questions, settings, client, run_folder, base_prompt, statistics, trace):
    print(f"\n\033[92m=== Running questions for model: {model_display_name} ({model_id}) ===\033[0m")
    model_run_folder = os.path.join(run_folder, model_id)
    os.makedirs(model_run_folder, exist_ok=True)

    for idx, (filename, question) in enumerate(questions, 1):
        # TODO: Remove evaluation of questions with match
        # Get the correct answer from the question, and remove it from the question text
        match = re.search(r'<(.*)>', question, re.DOTALL)

        # if settings.question_type == 'multiple_choice':
            # For multiple-choice questions, the correct answer is in brackets
            # correct_answer = "[" + match.group(1)  + "]"
        # elif settings.question_type == 'open_answer':
            # For open-answer questions, the correct answer is the text inside the brackets
        correct_answer = match.group(1)
        # Remove the correct answer from the question text
        if match:
            question = re.sub(r'<.*?>', '', question)
            question = question[:-1]

        prompt = base_prompt + question
        print(f"\n{prompt}")
        outputs = []
        # correct_answers = 0 # TODO: Remove evaluation of questions with match
        total_response_time = 0
        for run_idx in range(settings.num_runs_per_question):
            # Display experiment name, model name, question number, and iteration
            exp_name = os.path.basename(run_folder) if run_folder else 'N/A'
            q_str = f"{idx:2}"  # 2-character width for question number
            iter_str = f"{run_idx + 1:2}"  # 2-character width for iteration
            print(f"Experiment: {exp_name} | Model: {model_display_name} | Question: {q_str} | Iteration: {iter_str} | Answer: ", end="")

            opik_metadata = {
                "model_display_name": model_display_name,
                "model_id": model_id,
                "run_name": exp_name,
                "question_file": filename,
                "iteration": run_idx + 1,
                "correct_answer": correct_answer,
                "question_type": settings.question_type
            }

            start_time = time.time()

            answer = get_llm_response(settings, client, model_id, prompt)
            response_time = round(time.time() - start_time, 3)
            total_response_time += response_time

            answer_no_newlines = answer.replace('\n', '') if answer else ''
            answer_no_think = re.sub(r'<think>.*?</think>', '', answer_no_newlines, flags=re.DOTALL).strip() if answer_no_newlines else ''
            if settings.question_type == 'multiple_choice':
                # Keep only text matching the pattern [*], where * is a single character
                matches = re.findall(r'\[[^\]]\]', answer_no_think)
                answer_clean = ''.join(matches)                
            elif settings.question_type == 'open_answer':
                # For open-answer questions, keep the answer as is
                answer_clean = answer_no_think.strip()
                # # If the correct answer is in the answer, extract it
                # if correct_answer.lower() in answer_clean.lower():
                #     start = answer_clean.lower().find(correct_answer.lower())
                #     end = start + len(correct_answer)
                #     answer_clean = answer_clean[start:end]

            # is_correct = answer_clean.strip().lower() == correct_answer.strip().lower()
            
            # if is_correct:
            #     print(f"\033[92m{answer_clean}\033[0m")
            #     correct_answers += 1
            #     statistics.record_experiment(True, response_time)
            # else:
            #     print(f"\033[91m{answer_clean}\033[0m")
            #     statistics.record_experiment(False, response_time)
            
            # TODO: Remove evaluation of questions with match
            # TODO: Check the boolean param here:
            statistics.record_experiment(True, response_time)

            outputs.append({
                "question": question,
                "answer": answer_clean,
                "correct_answer": correct_answer,
                "raw_answer": answer_no_think,
                "model": model_display_name,
                "response_time (s)": response_time,
            })

            trace.span(
                name=f"q{idx}_r{run_idx + 1}",
                type="llm",
                model=model_display_name,
                input={
                    "question": question,
                },
                output={
                    "answer": answer_clean,
                    "correct_answer": correct_answer,
                    "raw_answer": answer_no_think,
                    "response_time (s)": response_time,
                },
                metadata=opik_metadata
            )

        # Calculate statistics
        # TODO: Remove evaluation of questions with match
        # accuracy = round(correct_answers / settings.num_runs_per_question, 2)
        avg_response_time = round(total_response_time / settings.num_runs_per_question, 3)
        # Insert statistics at the beginning of the outputs list
        outputs.insert(0, {
            # "num_correct": correct_answers, # TODO: Remove evaluation of questions with match
            # "accuracy": accuracy,
            "averaga_response_time (s)": avg_response_time,
        })

        out_file = os.path.join(model_run_folder, f"{settings.files['question_file_name']}{idx}.json")
        print(f"Writing to file: {out_file}")
        with open(out_file, "w") as f:
            json.dump(outputs, f, indent=2)

def main():
    settings = Settings()

    # Initialize client later: Ollama or OpenAI
    client = None
    
    # Access question dataset folder
    folder = open_dataset_folder(settings)

    # Ask user to choose between local or online models
    model_source = input("Do you want to run [l]ocal models or (o)nline models?: ").strip().upper()
    if model_source not in ['L', 'O', ''] or model_source == '':
        model_source = 'L'
    if model_source == 'L':
        models = get_local_models(settings.ollama_models)
        selected, run_all_models, client = select_local_model(models, settings.ollama['host'])        
    else:
        models = get_online_models(settings.openai_models)
        selected, run_all_models, client = select_online_model(models)
        models = [(name, mid) for name, mid, _, _, _ in models] # Remove base_url and api_key from models list

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
        statistics = Statistics()

        trace = opik_client.trace(
            name=f"{os.path.basename(run_folder)}_{model_id}",
            metadata={
                "model_id": model_id,
                "model_display_name": model_display_name,
                "run_name": os.path.basename(run_folder),
                "question_file": settings.files['question_file_name'],
                "question_type": settings.question_type,
                "num_runs_per_question": settings.num_runs_per_question,
                "model_source": "local" if model_source == 'L' else "online",
                "base_prompt": base_prompt
            }
        )

        run_questions_for_model(model_display_name, model_id, questions, settings, client, run_folder, base_prompt, statistics, trace)

        trace.end()
        
        statistics.print_statistics()
        statistics.save_statistics(os.path.join(run_folder, model_id, settings.files['stats_file_name']))

    print("\n=== All experiments completed ===")



if __name__ == "__main__":
    main()