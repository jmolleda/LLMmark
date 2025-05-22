import os
import re
import json
import requests
from settings import Settings
from ollama_client import OpenAIClient, ChatRunner

def get_models(settings):
    """Return a list of (display_name, model_id) tuples from config."""
    return [
        (m['display_name'], m['model_id'])
        for m in getattr(settings, 'models', [])
    ]

def get_questions_from_folder(folder, settings):
    """Read all question files from the given folder."""
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
    base_folder = settings.paths['base_experiments_folder']
    os.makedirs(base_folder, exist_ok=True)
    prefix = settings.folders['experiment_folder_name']
    existing = [d for d in os.listdir(base_folder) if d.startswith(prefix) and os.path.isdir(os.path.join(base_folder, d))]
    run_numbers = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    n = max(run_numbers, default=0) + 1
    run_folder = os.path.join(base_folder, f"{prefix}{n}")
    os.makedirs(run_folder)
    print(f"Created run experiment folder: {run_folder}")
    return run_folder

def get_llm_response(runner, model, prompt, stream):
    """Get the LLM response, streaming or not."""
    if stream:
        for chunk in runner.client.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True):
            print(chunk['message']['content'], end='', flush=True)
        print()
        return None
    response = runner.client.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=False)
    return response['message']['content']
    
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

def select_model(settings):
    """Prompt the user to select a model from config. If not installed, offer to pull it."""
    models = get_models(settings)
    print("Select a local model:")
    for idx, (name, _) in enumerate(models, 1):
        print(f"{idx}. {name}")
    choice = input("Enter model number [1]: ").strip()
    if not choice or not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        selected_model = models[0][1]
    else:
        selected_model = models[int(choice) - 1][1]
    print(f"Selected model: {selected_model}")

    if not is_model_installed(selected_model, settings.ollama['base_url']):
        print(f"{selected_model} is not available in ollama.")
        confirm = input(f"Do you want to pull '{selected_model}'? (Y/n): ").strip().lower()
        if confirm == '' or confirm == 'y':
            pull_model(selected_model, settings.ollama['base_url'])
            print(f"Pulling {selected_model}...")
            if is_model_installed(selected_model, settings.ollama['base_url']):
                print(f"{selected_model} successfully installed.")
            else:
                print(f"Failed to install {selected_model}.")
        else:
            print("Installation cancelled.")      

    return selected_model

def main():
    settings = Settings()
    
    folder = settings.folders['data_folder_name']
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        exit(1)
    print(f"Using folder: {folder}")

    selected_model = select_model(settings)

    stream_input = input("Do you want to run in streaming mode? (Y/n): ").strip().lower()
    stream = (stream_input == '' or stream_input == 'y')
    client = OpenAIClient(settings.ollama['base_url'], settings.ollama['api_key'])
    runner = ChatRunner(client)

    questions = get_questions_from_folder(folder, settings)

    run_folder = create_run_folder(settings) if not stream else None

    for idx, (filename, question) in enumerate(questions, 1):
        prompt = settings.default_prompt + question
        print(f"\n{prompt}")

        answer = get_llm_response(runner, selected_model, prompt, stream)

        if not stream:
            # Remove newlines and <think>...</think> blocks
            answer_no_newlines = answer.replace('\n', '') if answer else ''
            answer_cleaned = re.sub(r'<think>.*?</think>', '', answer_no_newlines, flags=re.DOTALL).strip() if answer_no_newlines else ''
            print(answer_cleaned)
            output = {
                "question": question,
                "answer": answer_cleaned
            }
            out_file = os.path.join(run_folder, f"{settings.files['question_file_name']}{idx}.json")
            print(f"Writing to file: {out_file}")
            with open(out_file, "w") as f:
                json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()