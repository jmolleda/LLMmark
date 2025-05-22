import os
import json
from ollama_client import OpenAIClient, ChatRunner
from models import Models

def get_questions_from_folder(folder):
    questions = []
    for filename in sorted(os.listdir(folder)):
        if filename.startswith("question_") and filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            print(f"Reading file: {file_path}")
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                continue
            with open(file_path, "r") as f:
                questions.append((filename, f.read().strip()))
    return questions

def create_run_folder(base_runs_folder="../data/runs"):
    os.makedirs(base_runs_folder, exist_ok=True)
    existing = [d for d in os.listdir(base_runs_folder) if d.startswith("run_") and os.path.isdir(os.path.join(base_runs_folder, d))]
    run_numbers = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    n = max(run_numbers, default=0) + 1
    run_folder = os.path.join(base_runs_folder, f"run_{n}")
    os.makedirs(run_folder)
    print(f"Created run experiment folder: {run_folder}")
    return run_folder

def get_llm_response(runner, model, prompt, stream):
    # Capture the LLM response as a string
    if stream:
        for chunk in runner.client.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True):
            print(chunk['message']['content'], end='', flush=True)
        print()
        return None
    else:
        response = runner.client.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=False)
        return response['message']['content']

def select_model():
    models = Models().get_models()
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

if __name__ == "__main__":
    folder = os.path.join("../data/questions", "exercises1")
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        exit(1)
    print(f"Using folder: {folder}")

    selected_model = select_model()

    stream_input = input("Do you want to run in streaming mode? (Y/n): ").strip().lower()
    stream = (stream_input == "" or stream_input == "y")
    client = OpenAIClient()
    runner = ChatRunner(client)

    questions = get_questions_from_folder(folder)

    if not stream:
        run_folder = create_run_folder()
    else:
        run_folder = None

    for idx, (filename, question) in enumerate(questions, 1):
        print(f"\n{question}")
        answer = get_llm_response(runner, model=selected_model, prompt=question, stream=stream)
        if not stream:
            answer_no_newlines = answer.replace('\n', '') if answer else ''
            print(answer_no_newlines)
            output = {
                "question": question,
                "answer": answer_no_newlines
            }
            out_file = os.path.join(run_folder, f"question_{idx}.json")
            print(f"Writing to file: {out_file}")
            with open(out_file, "w") as f:
                json.dump(output, f, indent=2)