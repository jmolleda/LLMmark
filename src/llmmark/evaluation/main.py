import os
import json
import time
import argparse
import logging
from collections import defaultdict
from ..prompt_generator import PromptGenerator
from ..settings import Settings
from ..logger_config import setup_logging
from rouge_score import rouge_scorer
from opik import Opik
from opik.evaluation.metrics import GEval
from opik.evaluation.models import LiteLLMChatModel
import litellm

logger = logging.getLogger(__name__)


def calculate_rouge(generated_text: str, reference_text: str) -> dict:
    if not generated_text or not reference_text:
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def check_contains(
    generated_text: str, required_phrases: list[str], case_sensitive: bool = False
) -> dict:
    if not generated_text or not required_phrases:
        return {"contains_percentage": 0.0, "all_contained": False}
    contained_count = sum(
        1
        for phrase in required_phrases
        if (phrase in generated_text)
        or (not case_sensitive and phrase.lower() in generated_text.lower())
    )
    total_phrases = len(required_phrases)
    percentage = (contained_count / total_phrases) * 100 if total_phrases > 0 else 0
    return {
        "contains_percentage": percentage,
        "all_contained": bool(contained_count == total_phrases),
    }


def calculate_geval(
    model_output: str,
    expected_answer: str,
    question: str,
    prompt_gen: PromptGenerator,
    settings: Settings,
) -> dict:
    time.sleep(2)
    try:
        judge_model_name = settings.evaluation.get(
            "judge_model", "gemini/gemini-1.5-flash-latest"
        )
        logger.info(f"  -> Using judge model: {judge_model_name}")
        eval_llm = LiteLLMChatModel(model_name=judge_model_name)
        eval_prompts = prompt_gen.get_evaluation_prompts()
        metric = GEval(
            model=eval_llm,
            task_introduction=eval_prompts["task_introduction"],
            evaluation_criteria=eval_prompts["evaluation_criteria"],
        )
        geval_input = f"Question: {question}\nExpected Answer: {expected_answer}\nModel Answer: {model_output}"
        score_result = metric.score(output=geval_input)
        if score_result is None:
            raise ValueError("GEval metric.score() returned None.")
        return {"geval_score": score_result.value, "geval_reason": score_result.reason}
    except Exception as e:
        logger.error(f"Error calculating GEval: {e}", exc_info=True)
        return {"geval_score": 0.0, "geval_reason": str(e)}


def load_model_answers(run_folder, model_id, question_file_prefix):
    model_path = os.path.join(run_folder, model_id)
    all_items = []
    if not os.path.isdir(model_path):
        logger.error(f"Model directory not found: {model_path}")
        return []
    for fname in sorted(os.listdir(model_path)):
        if fname.startswith(question_file_prefix) and fname.endswith(".json"):
            try:
                with open(os.path.join(model_path, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_items.append((fname, data[1:]))
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Could not parse file {fname}: {e}")
    return all_items


def group_by_question(data):
    grouped = defaultdict(list)
    for _, items in data:
        for entry in items:
            grouped[entry["question"]].append(entry)
    return grouped


def main():
    setup_logging()
    settings = Settings()
    opik_client = Opik(project_name="LLMmark_evaluation")

    parser = argparse.ArgumentParser(description="LLMmark evaluation runner.")
    parser.add_argument(
        "--run-folder",
        required=True,
        help="Name of the run folder to evaluate (e.g., 'run_083').",
    )
    parser.add_argument(
        "--model-id",
        default="all",
        help="Specific model ID (folder name) to evaluate, or 'all' for every model in the run.",
    )
    args = parser.parse_args()

    logger.info(
        f"--- Starting Evaluation for Run: '{args.run_folder}', Model(s): '{args.model_id}' ---",
        extra={"is_header": True},
    )
    prompt_gen = PromptGenerator(settings=settings)
    run_path = os.path.join(settings.experiments_folder, args.run_folder)

    if not os.path.isdir(run_path):
        logger.error(f"Run folder not found at: {run_path}")
        return

    available_models = sorted(
        [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    )
    selected_models_ids = (
        available_models if args.model_id.lower() == "all" else [args.model_id]
    )
    if args.model_id.lower() != "all" and args.model_id not in available_models:
        logger.error(
            f"Model ID '{args.model_id}' not found in run '{args.run_folder}'. Available: {available_models}"
        )
        return

    for model_id in selected_models_ids:
        raw_data = load_model_answers(
            run_path, model_id, settings.files["question_file_name"]
        )
        grouped = group_by_question(raw_data)
        logger.info(
            f"\n{'='*25} Evaluating Model: {model_id} {'='*25}",
            extra={"is_header": True},
        )
        results = []
        for q_idx, (question, answers) in enumerate(grouped.items(), 1):
            expected_answer = answers[0]["correct_answer"]
            logger.info(
                f"\n> Question {q_idx:02d}/{len(grouped):02d}: {question.strip()}"
            )
            logger.info(f"  Expected: {expected_answer.strip()}")
            trace = opik_client.trace(
                name=f"eval_{model_id}_q{q_idx}",
                metadata={
                    "model_id": model_id,
                    "run_name": args.run_folder,
                    "question_text": question.strip(),
                },
            )
            for run_idx, run in enumerate(answers[: settings.num_runs_per_question], 1):
                model_output = run["raw_answer"]
                logger.info(f"  [Run {run_idx:02d}] Answer: {model_output.strip()}")
                rouge_scores = calculate_rouge(model_output, expected_answer)
                contains_results = check_contains(
                    model_output,
                    [word.strip() for word in expected_answer.split() if word.strip()][
                        :5
                    ],
                )
                geval_scores = calculate_geval(
                    model_output, expected_answer, question, prompt_gen, settings
                )
                logger.info(
                    f"    Scores -> ROUGE: {rouge_scores['rougeL_f1']:.3f}, Contains: {contains_results['contains_percentage']:.1f}%, GEval: {geval_scores['geval_score']:.1f}"
                )
                span = trace.span(
                    name=f"run_{run_idx}",
                    input={
                        "model_output": model_output,
                        "expected_answer": expected_answer,
                    },
                    output={**rouge_scores, **contains_results, **geval_scores},
                    metadata={"run_iteration": run_idx},
                )
                scores_to_log = [
                    {"id": span.id, "name": name, "value": float(value)}
                    for name, value in {**rouge_scores, **contains_results}.items()
                ]
                scores_to_log.append(
                    {
                        "id": span.id,
                        "name": "geval_score",
                        "value": geval_scores["geval_score"],
                        "reason": geval_scores.get("geval_reason", ""),
                    }
                )
                opik_client.log_spans_feedback_scores(scores=scores_to_log)
                results.append(
                    {
                        "question": question,
                        "run": run_idx,
                        "model_output": model_output,
                        "expected_answer": expected_answer,
                        "rouge_scores": rouge_scores,
                        "contains_results": contains_results,
                        "geval_scores": geval_scores,
                    }
                )
            trace.end()
        output_path = os.path.join(run_path, model_id, settings.evaluation_output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results for {model_id} saved to: {output_path}")
    logger.info("\n=== All evaluations completed ===", extra={"is_header": True})


if __name__ == "__main__":
    main()
