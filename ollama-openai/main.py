import os
from ollama_client import OpenAIClient, ChatRunner

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
                questions.append(f.read().strip())
    return questions

if __name__ == "__main__":
    folder = os.path.join("../data/questions", "exercises1")
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        exit(1)
    print(f"Using folder: {folder}")

    stream_input = input("Do you want to run in streaming mode? (Y/n): ").strip().lower()
    stream = (stream_input == "" or stream_input == "y")
    client = OpenAIClient()
    runner = ChatRunner(client)

    questions = get_questions_from_folder(folder)
    for idx, question in enumerate(questions, 1):
        print(f"\n{question}")
        runner.run(
            model="gemma3:1b",
            prompt=question,
            stream=stream
        )