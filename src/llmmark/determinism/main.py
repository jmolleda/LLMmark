import os
import re
import json
import time
import requests
from ..settings import Settings
from ..statistics import Statistics
from ..prompt_generator import PromptGenerator
from ..clients.openai_client import OpenAIClient
from ..clients.ollama_client import OllamaClient
from opik import Opik

opik_client = Opik(project_name="LLMmark_determinism")

def open_dataset_folder(settings):
    """Open the dataset folder based on the settings.

    Args:
        settings (Settings): The settings object containing configuration.

    Returns:
        tuple: The path to the selected folder, the language code and the selected folder path.
    """
    base_folder = settings.folders['data_folder_name']
    base_folder = os.path.join(base_folder, settings.question_type)
    
    if not os.path.exists(base_folder) or not os.path.isdir(base_folder):
        print(f"Base data folder '{base_folder}' does not exist.")
        exit(1)

    try:
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

    choice = input(f"Enter the number of the folder (default [1] for {exercise_folders[0]}): ").strip()
    
    selected_index = 0
    if choice.isdigit() and 1 <= int(choice) <= len(exercise_folders):
        selected_index = int(choice) - 1
    elif choice:
        print(f"Invalid choice. Using default folder: {exercise_folders[0]}")

    selected_folder_name = exercise_folders[selected_index]
    selected_folder_path = os.path.join(base_folder, selected_folder_name)
    print(f"Selected folder: \033[92m{selected_folder_name}\033[0m")

    potential_languages = {"en": "English", "es": "Spanish"}
    found_languages = [lang for lang in potential_languages if os.path.isdir(os.path.join(selected_folder_path, lang))]

    if not found_languages:
        print(f"No language subfolders found. Using default language from config: {settings.language}")
        return selected_folder_path, settings.language

    print("\nLanguage options found. Please select one:")
    for idx, lang_code in enumerate(found_languages, 1):
        print(f"  {idx}. {potential_languages[lang_code]}")

    default_lang_display = potential_languages.get(settings.language, "the first option")
    lang_choice = input(f"Enter language number (default [1] for {default_lang_display}): ").strip()

    default_lang_idx = 0
    if settings.language in found_languages:
        default_lang_idx = found_languages.index(settings.language)

    selected_lang_idx = default_lang_idx
    if lang_choice.isdigit() and 1 <= int(lang_choice) <= len(found_languages):
        selected_lang_idx = int(lang_choice) - 1
    elif lang_choice:
        print("Invalid language choice. Using default.")

    final_lang_code = found_languages[selected_lang_idx]
    final_path = os.path.join(selected_folder_path, final_lang_code)
    
    print(f"Selected language: \033[92m{potential_languages[final_lang_code]}\033[0m")
    
    return final_path, final_lang_code, selected_folder_path

def get_local_models(models):
    """Get local model IDs and display names.

    Args:
        models (list): List of model dictionaries.

    Returns:
        list: A list of tuples containing display names and model IDs.
    """
    return [(model['display_name'], model['model_id']) for model in models]

def get_online_models(models):
    """Get online model IDs and their details.

    Args:
        models (list): List of model dictionaries.

    Returns:
        list: A list of tuples containing display names, model IDs, base URLs, API keys, and max requests per minute.
    """
    return [(model['display_name'], model['model_id'], model['base_url'], model['api_key'], model['max_requests_per_minute']) for model in models]

def get_questions_from_folder(folder, settings):
    """Get questions from a folder based on the settings.

    Args:
        folder (str): The path to the folder containing question files.
        settings (Settings): The settings object containing configuration.

    Returns:
        list: A list of tuples containing the filename and file content.
    """
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
            with open(file_path, "r", encoding='utf-8') as f:
                questions.append((filename, f.read().strip()))
    return questions

def create_run_folder(settings):
    """Creates a new run folder for experiments.

    Args:
        settings (Settings): The settings object containing configuration.

    Returns:
        str: The path to the created run folder.
    """
    base_folder = settings.folders['base_experiments_folder']
    os.makedirs(base_folder, exist_ok=True)
    prefix = settings.folders['experiment_folder_name']
    existing = [d for d in os.listdir(base_folder) if d.startswith(prefix) and os.path.isdir(os.path.join(base_folder, d))]
    run_numbers = [int(d.split('_')[1]) for d in existing if len(d.split('_')) > 1 and d.split('_')[1].isdigit()]
    n = max(run_numbers, default=0) + 1
    run_folder_name = f"{prefix}{n:03d}"
    run_folder = os.path.join(base_folder, run_folder_name)
    os.makedirs(run_folder)
    print(f"Created run experiment folder: {run_folder}")
    return run_folder

def get_llm_response(settings, client, model, system_prompt, user_prompt, temperature=0.7):
    """Gets the response from the LLM.

    Args:
        settings (Settings): The settings object containing configuration.
        client (OllamaClient): The client instance for making requests.
        model (str): The model ID to use for the request.
        system_prompt (str): The system prompt to provide context.
        user_prompt (str): The user prompt containing the question.
        temperature (float, optional): The temperature for the response generation. Defaults to 0.7.

    Returns:
        dict: The response from the LLM.
    """
    return client.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

def is_model_installed(model_name: str, ollama_base_url) -> bool:
    """Check if a model is installed in Ollama.
    Args:
        model_name (str): The name of the model to check.
        ollama_base_url (str): The base URL of the Ollama server.
    Returns:
        bool: True if the model is installed, False otherwise.
    """
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(model.get("name", "").startswith(model_name) for model in models)
    except requests.exceptions.RequestException as e:
        print(f"Error checking model list: {e}")
        return False

def pull_model(model_name: str, ollama_base_url) -> None:
    """Pulls a model from the Ollama server.

    Args:
        model_name (str): The name of the model to pull.
        ollama_base_url (str): The base URL of the Ollama server.
    """
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
    """Select a local model for inference.

    Args:
        models (list): List of available models.
        host (str): The host URL for the Ollama client.

    Returns:
        tuple: The selected model ID and the client instance.
    """
    client = OllamaClient(host)
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
    """Select an online model for inference.

    Args:
        models (list): List of available online models.

    Returns:
        tuple: The selected model ID and the client instance.
    """
    print("Available online models:")
    for idx, (name, _, _, _, _) in enumerate(models, 1):
        print(f"{idx}. {name}")

    choice = input("Enter model number (default [1]): ").strip().lower()
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(models) else 0

    selected_model = models[idx][1]
    base_url = models[idx][2]
    api_key_env_var = models[idx][3]
    api_key = os.environ.get(api_key_env_var)
    max_requests_per_minute = models[idx][4]
    
    if not api_key:
        print(f"Warning: Environment variable '{api_key_env_var}' for API key not set.")

    print(f"Selected model: \033[92m{selected_model}\033[0m")
    client = OpenAIClient(api_key=api_key, base_url=base_url, max_requests_per_minute=max_requests_per_minute)
    return selected_model, False, client

def select_question_type(settings):
    """Selects the question type to use for inference.

    Args:
        settings (Settings): The settings object containing configuration.

    Returns:
        str: The selected question type key.
    """
    question_types_list = settings.question_types
    if not question_types_list:
        print("No question types defined in the settings file.")
        exit(1)

    available_types = {k: v for item in question_types_list for k, v in item.items()}
    options = list(available_types.keys())

    print("\nPlease select the question type to run:")
    for idx, key in enumerate(options, 1):
        print(f"  {idx}. {available_types[key]}")

    choice = input(f"Enter number (default [1] for {available_types[options[0]]}): ").strip()
    selected_index = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(options) else 0
    selected_type_key = options[selected_index]
    
    print(f"Question type selected: \033[92m{available_types[selected_type_key]}\033[0m")
    return selected_type_key

def run_questions_for_model(model_display_name, model_id, questions, settings, client, run_folder, prompt_gen, question_type_key, statistics, trace, few_shot_examples="", information=""):
    """Runs a series of questions against a specified model.

    Args:
        model_display_name (str): The display name of the model.
        model_id (str): The ID of the model.
        questions (list): A list of tuples containing question filenames and their content.
        settings (Settings): The settings object containing configuration.
        client (OllamaClient): The client instance for making requests.
        run_folder (str): The path to the folder for saving run results.
        prompt_gen (PromptGenerator): The prompt generator instance.
        question_type_key (str): The key for the question type to use.
        statistics (dict): A dictionary to store statistics about the runs.
        trace (list): A list to store traces of the runs.
        few_shot_examples (str): Few-shot examples to include in the prompts, if any.
        information (str): Additional information to include in the prompts, if any.
    """
    print(f"\n\033[92m=== Running questions for model: {model_display_name} ({model_id}) ===\033[0m")
    model_run_folder = os.path.join(run_folder, model_id.replace("/", "_")) # Sanitize model_id for folder name
    os.makedirs(model_run_folder, exist_ok=True)

    for idx, (filename, question) in enumerate(questions, 1):
        match = re.search(r'<(.*)>', question, re.DOTALL)
        correct_answer = match.group(1).strip() if match else ""
        if match:
            question = re.sub(r'<.*?>', '', question).strip()
        
        format_args = {
            "question": question,
            "example": few_shot_examples,
            "info": information
        }
        
        prompting_tech = settings.prompting_technique
        language = settings.language
        
        question_type_key = settings.question_type
        valid_keys = [list(item.keys())[0] for item in settings.question_types] 
        if question_type_key not in valid_keys:
            print(f"Question type '{question_type_key}' not found in settings.")
            exit(1)
            
            
        system_prompt = prompt_gen.get_system_prompt(
            prompt_key=prompting_tech,
            question_type=settings.question_type,
            **format_args
        )
        
        user_prompt = prompt_gen.get_user_prompt(
            prompt_key=prompting_tech,
            **format_args
        )
        
        print(f"Using prompting technique: \033[92m{prompting_tech}\033[0m for language: \033[92m{language}\033[0m")
        print(f"- Using system prompt: \033[92m{system_prompt}\033[0m")
        print(f"- Using user prompt: \033[92m{user_prompt}\033[0m")
        
     
        outputs = []
        total_response_time = 0
        for run_idx in range(settings.num_runs_per_question):
            exp_name = os.path.basename(run_folder)
            q_str, iter_str = f"{idx:2}", f"{run_idx + 1:2}"
            print(f"Experiment: {exp_name} | Model: {model_display_name} | Question: {q_str} | Iteration: {iter_str} | Answer: ", end="")
            
            opik_metadata = {
                "model_display_name": model_display_name, "model_id": model_id, "run_name": exp_name,
                "question_file": filename, "iteration": run_idx + 1, "correct_answer": correct_answer,
                "question_type": settings.question_type
            }
            start_time = time.time()
            
            answer = get_llm_response(settings, client, model_id, system_prompt, user_prompt, settings.temperature) or ""
            response_time = round(time.time() - start_time, 3)
            total_response_time += response_time

            answer_no_newlines = answer.replace('\n', ' ')
            answer_no_think = re.sub(r'<think>.*?</think>', '', answer_no_newlines, flags=re.DOTALL).strip()
            
            answer_clean = answer_no_think
            if settings.question_type == 'multiple_choice_questions':
                matches = re.findall(r'\[[a-f]\]', answer_no_think)
                answer_clean = ''.join(matches)

            statistics.record_experiment(True, response_time)
            outputs.append({
                "question": question, "answer": answer_clean, "correct_answer": correct_answer,
                "raw_answer": answer_no_think, "model": model_display_name, "response_time (s)": response_time,
            })
            
            
            trace.span(
                name=f"q{idx}_r{run_idx + 1}", type="llm", model=model_display_name,
                input={"question": question},
                output={"answer": answer_clean, "correct_answer": correct_answer, "raw_answer": answer_no_think, "response_time (s)": response_time},
                metadata=opik_metadata
            )

        avg_response_time = round(total_response_time / settings.num_runs_per_question, 3)
        outputs.insert(0, {"average_response_time (s)": avg_response_time})

        out_file = os.path.join(model_run_folder, f"{settings.files['question_file_name']}{idx}.json")
        with open(out_file, "w", encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def main():
    settings = Settings()
    
    # Question type selection
    question_type = select_question_type(settings)
    settings.question_type = question_type
    
    # Language selection
    folder, selected_language, selected_folder_path = open_dataset_folder(settings)
    
    settings.language = selected_language
    print(f"Language for this run set to: \033[92m{settings.language}\033[0m")

    # Prompt generator to create prompts
    prompt_gen = PromptGenerator(settings=settings)

    # Model source selection
    model_source = input("Do you want to run [l]ocal models or (o)nline models? (default: local): ").strip().upper()
    if model_source not in ['L', 'O', ''] or model_source == '': model_source = 'L'
    
    run_all_models = False
    if model_source == 'L':
        models_list = get_local_models(settings.ollama_models)
        selected, run_all_models, client = select_local_model(models_list, settings.ollama['host'])
    else:
        models_list = get_online_models(settings.openai_models)
        selected, run_all_models, client = select_online_model(models_list)
        models_list = [(name, mid) for name, mid, _, _, _ in models_list]

    if run_all_models:
        selected_models = [(name, mid) for name, mid in models_list if mid in set(selected)]
    else:
        selected_models = [(next((name for name, mid in models_list if mid == selected), selected), selected)]

    run_folder = create_run_folder(settings)
    questions = get_questions_from_folder(folder, settings)
    
    few_shot_examples = prompt_gen.get_few_shot_examples(selected_folder_path)
    
    information = prompt_gen.get_information(selected_folder_path)

    for model_display_name, model_id in selected_models:
        statistics = Statistics()

        prompting_tech = settings.prompting_technique
        
        trace_metadata = {
            "model_id": model_id,
            "model_display_name": model_display_name,
            "run_name": os.path.basename(run_folder),
            "language": settings.language,
            "prompting_tech": prompting_tech,
            "num_runs_per_question": settings.num_runs_per_question,
            "model_source": "local" if model_source == 'L' else "online",
            "temperature": settings.temperature,
            "top-p": settings.top_p
        }
        
        opik_prompt_obj = prompt_gen.get_opik_prompt_object(prompting_tech)

        trace = opik_client.trace(
            name=f"{os.path.basename(run_folder)}_{model_id}",
            metadata=trace_metadata,
            prompt=opik_prompt_obj,
            tags={settings.question_type, settings.language, prompting_tech, "local" if model_source == 'L' else "online", model_id, settings.temperature}
        )
        
        
        run_questions_for_model(model_display_name, model_id, questions, settings, client, run_folder, prompt_gen, question_type, statistics, trace, few_shot_examples, information)

        trace.end()
        
        # statistics.print_statistics()
        # statistics.save_statistics(os.path.join(run_folder, model_id, settings.files['stats_file_name']))

    print("\n=== All experiments completed ===")

if __name__ == "__main__":
    main()