import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from settings import Settings
from main import (
    get_models,
    get_questions_from_folder,
    create_run_folder
)

def test_get_models():
    settings = Settings()
    models = get_models(settings.models)
    assert isinstance(models, list)
    for name, model_id in models:
        assert isinstance(name, str)
        assert isinstance(model_id, str)

def test_get_questions_from_folder(tmp_path: Path):
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

def test_create_run_folder(tmp_path: Path):
    settings = Settings()
    settings.folders['base_experiments_folder'] = str(tmp_path)
    settings.folders['experiment_folder_name'] = 'run_'
    run_folder = create_run_folder(settings)
    assert os.path.exists(run_folder)
    assert os.path.isdir(run_folder)
    # Should create run_1 if none exists
    assert run_folder.endswith('run_1')
    # Create another, should be run_2
    run_folder2 = create_run_folder(settings)
    assert run_folder2.endswith('run_2')
