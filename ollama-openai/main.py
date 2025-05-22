import os
import json
from settings import Settings
from ollama_client import OpenAIClient, ChatRunner

def get_models(settings):
    """Return a list of (display_name, model_id) tuples from config."""
    if hasattr(settings, 'models'):
        return [(m['display_name'], m['model_id']) for m in settings.models]
    return []

def get_questions_from_folder(folder, settings):
    """Read all question files from the given folder."""
    questions = []
    for filename in sorted(os.listdir(folder)):
        if filename.startswith(settings.files['question_file_name']) and filename.endswith(settings.files['question_file_extension']):
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
    base_experiments_folder = settings.paths['base_experiments_folder']
    os.makedirs(base_experiments_folder, exist_ok=True)
    experiment_prefix = settings.folders['experiment_folder_name']
    existing = [d for d in os.listdir(base_experiments_folder) if d.startswith(experiment_prefix) and os.path.isdir(os.path.join(base_experiments_folder, d))]
    run_numbers = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    n = max(run_numbers, default=0) + 1
    run_folder = os.path.join(base_experiments_folder, f"{experiment_prefix}{n}")
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
    else:
        response = runner.client.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=False)
        return response['message']['content']

def select_model(settings):
    """Prompt the user to select a model from config."""
    models = get_models(settings)
    print("Select a local model:")
    for idx, (name, _) in enumerate(models, 1):
        print(f"{idx}. {name}")
    choice = input(f"Enter model number [1]: ").strip()
    if choice == "" or not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        selected_model = models[0][1]
    else:
        selected_model = models[int(choice)-1][1]
    print(f"Selected model: {selected_model}")
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
    stream = (stream_input == "" or stream_input == "y")
    client = OpenAIClient()
    runner = ChatRunner(client)

    questions = get_questions_from_folder(folder, settings)

    run_folder = None
    if not stream:
        run_folder = create_run_folder(settings)

    for idx, (filename, question) in enumerate(questions, 1):
        prompt = settings.default_prompt + question
        print(f"\n{prompt}")

        answer = get_llm_response(runner, selected_model, prompt, stream)

        if not stream:
            answer_no_newlines = answer.replace('\n', '') if answer else ''
            print(answer_no_newlines)
            output = {
                "question": question,
                "answer": answer_no_newlines
            }
            out_file = os.path.join(run_folder, f"{settings.files['question_file_name']}{idx}.json")
            print(f"Writing to file: {out_file}")
            with open(out_file, "w") as f:
                json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()