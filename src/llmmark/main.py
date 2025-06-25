import os
import re
import json
import time
import requests
import argparse
import logging
from .settings import Settings
from .statistics import Statistics
from .prompt_generator import PromptGenerator
from .clients.openai_client import OpenAIClient
from .clients.ollama_client import OllamaClient
from .logger_config import setup_logging
from opik import Opik

logger = logging.getLogger(__name__)


def get_local_models(models):
    return [(model["display_name"], model["model_id"]) for model in models]


def get_online_models(models):
    return [
        (
            model["display_name"],
            model["model_id"],
            model["base_url"],
            model["api_key"],
            model["max_requests_per_minute"],
        )
        for model in models
    ]


def get_questions_from_folder(folder, settings):
    questions, prefix, ext = (
        [],
        settings.files["question_file_name"],
        settings.files["question_file_extension"],
    )
    for filename in sorted(os.listdir(folder)):
        if filename.startswith(prefix) and filename.endswith(ext):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                questions.append((filename, f.read().strip()))
    if not questions:
        logger.error(f"No questions found in '{folder}'.")
        exit(1)
    return questions


def create_run_folder(settings):
    base_folder = settings.folders["base_experiments_folder"]
    os.makedirs(base_folder, exist_ok=True)
    prefix = settings.folders["experiment_folder_name"]
    existing = [
        d
        for d in os.listdir(base_folder)
        if d.startswith(prefix) and os.path.isdir(os.path.join(base_folder, d))
    ]
    run_numbers = [
        int(d.split("_")[1])
        for d in existing
        if len(d.split("_")) > 1 and d.split("_")[1].isdigit()
    ]
    n = max(run_numbers, default=0) + 1
    run_folder_name = f"{prefix}{n:03d}"
    run_folder = os.path.join(base_folder, run_folder_name)
    os.makedirs(run_folder)
    return run_folder


def get_llm_response(settings, client, model, system_prompt, user_prompt):
    return client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.temperature,
    )


def is_model_installed(model_name: str, ollama_base_url) -> bool:
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status()
        return any(
            model.get("name", "").startswith(model_name)
            for model in response.json().get("models", [])
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking Ollama model list: {e}")
        return False


def pull_model(model_name: str, ollama_base_url) -> None:
    logger.info(
        f"Attempting to pull model '{model_name}' from Ollama. This may take a while..."
    )
    try:
        response = requests.post(
            f"{ollama_base_url}/api/pull", json={"name": model_name}, stream=True
        )
        for line in response.iter_lines():
            if line:
                status = json.loads(line.decode("utf-8"))
                if "status" in status:
                    log_line = f"  Pulling {model_name}: {status['status']}"
                    if "total" in status and "completed" in status:
                        progress = (
                            status.get("completed", 0) / status.get("total", 1) * 100
                        )
                        print(f"\r{log_line} | {progress:.2f}%", end="")
        print()
        logger.info(f"Finished pulling {model_name}.")
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Error pulling model '{model_name}': {e}")


def run_questions_for_model(
    model_display_name,
    model_id,
    questions,
    settings,
    client,
    run_folder,
    prompt_gen,
    statistics,
    trace,
    few_shot_examples="",
    information="",
    reasoning_info="",
):
    logger.info(
        f"\n{'='*25} Running Model: {model_display_name} ({model_id}) {'='*25}",
        extra={"is_header": True},
    )
    model_run_folder = os.path.join(run_folder, model_id.replace("/", "_"))
    os.makedirs(model_run_folder, exist_ok=True)
    for idx, (filename, question) in enumerate(questions, 1):
        match = re.search(r"<(.*)>", question, re.DOTALL)
        correct_answer = match.group(1).strip() if match else ""
        if match:
            question = re.sub(r"<.*?>", "", question).strip()
        logger.info(f"\n> Question {idx:02d}/{len(questions):02d}: {question}")
        format_args = {
            "question": question,
            "example": few_shot_examples,
            "info": information,
            "reasoning_instructions": reasoning_info,
        }
        system_prompt = prompt_gen.get_system_prompt(
            prompt_key=settings.prompting_technique,
            question_type=settings.question_type,
            **format_args,
        )
        user_prompt = prompt_gen.get_user_prompt(
            prompt_key=settings.prompting_technique, **format_args
        )
        outputs, total_response_time = [], 0
        for run_idx in range(settings.num_runs_per_question):
            start_time = time.time()
            answer = (
                get_llm_response(settings, client, model_id, system_prompt, user_prompt)
                or ""
            )
            response_time = round(time.time() - start_time, 3)
            total_response_time += response_time
            answer_no_think = re.sub(
                r"<think>.*?</think>", "", answer.replace("\n", " "), flags=re.DOTALL
            ).strip()
            answer_clean = (
                "".join(re.findall(r"\[[a-f]\]", answer_no_think))
                if settings.question_type == "multiple_choice"
                else answer_no_think
            )
            logger.info(
                f"  [Run {run_idx + 1:02d}/{settings.num_runs_per_question:02d}] Time: {response_time:>5.2f}s | Answer: {answer_clean}"
            )
            statistics.record_experiment(True, response_time)
            outputs.append(
                {
                    "question": question,
                    "answer": answer_clean,
                    "correct_answer": correct_answer,
                    "raw_answer": answer,
                    "model": model_display_name,
                    "response_time (s)": response_time,
                }
            )
            opik_metadata = {
                "model_display_name": model_display_name,
                "model_id": model_id,
                "run_name": os.path.basename(run_folder),
                "question_file": filename,
                "iteration": run_idx + 1,
                "correct_answer": correct_answer,
                "question_type": settings.question_type,
            }
            trace.span(
                name=f"q{idx}_r{run_idx + 1}",
                type="llm",
                model=model_display_name,
                input={"question": question},
                output={
                    "answer": answer_clean,
                    "correct_answer": correct_answer,
                    "raw_answer": answer_no_think,
                    "response_time (s)": response_time,
                },
                metadata=opik_metadata,
            )
        avg_response_time = round(
            total_response_time / settings.num_runs_per_question, 3
        )
        outputs.insert(0, {"average_response_time (s)": avg_response_time})
        out_file = os.path.join(
            model_run_folder, f"{settings.files['question_file_name']}{idx:02d}.json"
        )
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


def main():
    setup_logging()
    settings = Settings()
    opik_client = Opik(project_name="LLMmark_response_generation")
    parser = argparse.ArgumentParser(description="LLMmark benchmark runner.")
    parser.add_argument(
        "-qt",
        "--question-type",
        type=str,
        default=settings.question_type,
        choices=[list(item.keys())[0] for item in settings.question_types],
        help="The type of questions to run.",
    )
    parser.add_argument(
        "-ef",
        "--exercise-folder",
        required=True,
        type=str,
        help="Subfolder name inside data/questions/[question_type]/ containing the questions.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=settings.language,
        choices=["en", "es"],
        help="Language of the questions.",
    )
    parser.add_argument(
        "-ms",
        "--model-source",
        type=str,
        default="local",
        choices=["local", "online"],
        help="Source of the models to run.",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="all",
        help="ID of the model to run, or 'all' to run all models from the source.",
    )
    parser.add_argument(
        "-pt",
        "--prompting-technique",
        type=str,
        default=settings.prompting_technique,
        choices=[list(item.keys())[0] for item in settings.prompting_techniques],
        help="The prompting technique to use.",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=settings.num_runs_per_question,
        help="Number of times to run each question.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=settings.temperature,
        help="Temperature for model generation.",
    )
    parser.add_argument(
        "-p",
        "--top-p",
        type=float,
        default=settings.top_p,
        help="Top-p for model generation.",
    )
    args = parser.parse_args()

    settings.question_type, settings.language, settings.prompting_technique = (
        args.question_type,
        args.language,
        args.prompting_technique,
    )
    settings.num_runs_per_question, settings.temperature, settings.top_p = (
        args.num_runs,
        args.temperature,
        args.top_p,
    )
    logger.info("--- LLMmark Run Configuration ---", extra={"is_header": True})
    for key, value in vars(args).items():
        logger.info(f"  {key:<22}: {value}")
    logger.info("---------------------------------", extra={"is_header": True})

    base_data_folder = settings.folders["data_folder_name"]
    questions_folder = os.path.join(
        base_data_folder, args.question_type, args.exercise_folder, args.language
    )
    exercise_path = os.path.join(
        base_data_folder, args.question_type, args.exercise_folder
    )
    if not os.path.isdir(questions_folder):
        logger.error(f"Data folder does not exist: {questions_folder}")
        exit(1)
    prompt_gen = PromptGenerator(settings=settings)

    client, selected_models_info = None, []
    if args.model_source == "local":
        models_list = get_local_models(settings.ollama_models)
        client = OllamaClient(
            host=settings.ollama["host"],
            top_p=settings.top_p,
            timeout=settings.ollama.get("timeout", 120),
        )
        model_ids_to_run = (
            [m[1] for m in models_list]
            if args.model_id.lower() == "all"
            else [args.model_id]
        )
        selected_models_info = [m for m in models_list if m[1] in model_ids_to_run]
    else:
        models_list = get_online_models(settings.openai_models)
        selected_models_info = (
            models_list
            if args.model_id.lower() == "all"
            else [m for m in models_list if m[1] == args.model_id]
        )
    if not selected_models_info:
        logger.error(
            f"Model ID '{args.model_id}' not found for source '{args.model_source}'."
        )
        exit(1)

    run_folder = create_run_folder(settings)
    logger.info(f"Created run folder: {run_folder}")
    questions = get_questions_from_folder(questions_folder, settings)
    logger.info(f"Found {len(questions)} questions to run.")
    few_shot_examples = prompt_gen.get_few_shot_examples(exercise_path)
    reasoning_info = prompt_gen.get_reasoning_information()
    information = prompt_gen.get_information(exercise_path)

    for model_info in selected_models_info:
        if args.model_source == "local":
            model_display_name, model_id = model_info
            if not is_model_installed(model_id, settings.ollama["host"]):
                logger.warning(
                    f"Model '{model_id}' is not installed locally. Pulling..."
                )
                pull_model(model_id, settings.ollama["host"])
                if not is_model_installed(model_id, settings.ollama["host"]):
                    logger.error(f"Failed to install model '{model_id}'. Skipping.")
                    continue
        else:
            model_display_name, model_id, base_url, api_key_env, max_rpm = model_info
            api_key = os.environ.get(api_key_env)
            if not api_key:
                logger.warning(
                    f"Env var '{api_key_env}' not set for model '{model_id}'. Skipping."
                )
                continue
            client = OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                max_requests_per_minute=max_rpm,
                top_p=settings.top_p,
            )

        statistics = Statistics()
        trace_metadata = {
            "model_id": model_id,
            "model_display_name": model_display_name,
            "run_name": os.path.basename(run_folder),
            "language": settings.language,
            "prompting_tech": settings.prompting_technique,
            "num_runs_per_question": settings.num_runs_per_question,
            "model_source": args.model_source,
            "temperature": settings.temperature,
            "top-p": settings.top_p,
            "exercise": args.exercise_folder,
        }
        opik_prompt_obj = prompt_gen.get_opik_prompt_object(
            settings.prompting_technique
        )
        trace = opik_client.trace(
            name=f"{os.path.basename(run_folder)}_{model_id}",
            metadata=trace_metadata,
            prompt=opik_prompt_obj,
            tags={
                settings.question_type,
                settings.language,
                settings.prompting_technique,
                args.model_source,
                model_id,
                str(settings.temperature),
                args.exercise_folder
            },
        )
        run_questions_for_model(
            model_display_name,
            model_id,
            questions,
            settings,
            client,
            run_folder,
            prompt_gen,
            statistics,
            trace,
            few_shot_examples,
            information,
            reasoning_info,
        )
        trace.end()
        stats_path = os.path.join(
            run_folder, model_id.replace("/", "_"), settings.files["stats_file_name"]
        )
        statistics.save_statistics(stats_path)
        statistics.log_statistics()
    logger.info("\n=== All experiments completed ===", extra={"is_header": True})


if __name__ == "__main__":
    main()
