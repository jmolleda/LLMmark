import os
import tempfile
import shutil
import json
import pytest
from ollama_openai.settings import Settings
from ollama_openai.main import (
    get_models,
    get_questions_from_folder,
    create_run_folder,
    get_llm_response,
    select_model
)

class DummyRunner:
    def __init__(self, response):
        self.response = response
        self.client = self
    def chat(self, model, messages, stream=True):
        if stream:
            for c in self.response:
                yield {'message': {'content': c}}
        else:
            return {'message': {'content': self.response}}

def test_get_models():
    settings = Settings()
    models = get_models(settings)
    assert isinstance(models, list)
    for name, model_id in models:
        assert isinstance(name, str)
        assert isinstance(model_id, str)

def test_get_questions_from_folder(tmp_path):
    # Setup
    settings = Settings()
    settings.files['question_file_name'] = 'question_'
    settings.files['question_file_extension'] = '.txt'
    folder = tmp_path / "questions"
    folder.mkdir()
    (folder / "question_1.txt").write_text("What is 2+2?")
    (folder / "question_2.txt").write_text("What is the capital of France?")
    questions = get_questions_from_folder(str(folder), settings)
    assert len(questions) == 2
    assert questions[0][1] == "What is 2+2?"
    assert questions[1][1] == "What is the capital of France?"

def test_create_run_folder(tmp_path, monkeypatch):
    settings = Settings()
    settings.paths['base_experiments_folder'] = str(tmp_path)
    settings.folders['experiment_folder_name'] = 'run_'
    run_folder = create_run_folder(settings)
    assert os.path.exists(run_folder)
    assert os.path.isdir(run_folder)
    # Should create run_1 if none exists
    assert run_folder.endswith('run_1')
    # Create another, should be run_2
    run_folder2 = create_run_folder(settings)
    assert run_folder2.endswith('run_2')

def test_get_llm_response_stream():
    runner = DummyRunner(["Hello", " world!"])
    # Should print to stdout, but we just check it returns None
    result = get_llm_response(runner, "model", "prompt", stream=True)
    assert result is None

def test_get_llm_response_no_stream():
    runner = DummyRunner("42")
    result = get_llm_response(runner, "model", "prompt", stream=False)
    assert result == "42"

# select_model is interactive, so we do not test it directly here.
# You can use unittest.mock to patch input() if needed for more advanced tests.
