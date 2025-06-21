import os
import json
import time
import yaml
from collections import defaultdict
from llmmark.prompt_generator import PromptGenerator
from ..settings import Settings
from rouge_score import rouge_scorer
from opik import Opik
from opik.evaluation.metrics import GEval
from opik.evaluation.models import LiteLLMChatModel
import litellm

opik_client = Opik(project_name="LLMmark_evaluation")

def calculate_rouge(generated_text: str, reference_text: str) -> dict:
    if not generated_text or not reference_text:
        return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)

    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure,
    }

def check_contains(generated_text: str, required_phrases: list[str], case_sensitive: bool = False) -> dict:
    if not generated_text or not required_phrases:
        return {"contains_percentage": 0.0, "all_contained": 0}

    contained_count = 0
    total_phrases = len(required_phrases)
    
    for phrase in required_phrases:
        if case_sensitive:
            is_contained = phrase in generated_text
        else:
            is_contained = phrase.lower() in generated_text.lower()
        if is_contained:
            contained_count += 1

    percentage = (contained_count / total_phrases) * 100 if total_phrases > 0 else 0
    all_contained = (contained_count == total_phrases)

    return {
        "contains_percentage": percentage,
        "all_contained": bool(all_contained)
    }
    
def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    

def calculate_geval(model_output: str, expected_answer: str, question: str, prompt_gen: PromptGenerator) -> dict:
    # Quota limits for gemini
    # time.sleep(15)
    try:
        # Gemini 2.0 flash lite as model for LLM-as-a-judge
        eval_llm = LiteLLMChatModel(model_name="gemini/gemini-2.0-flash-lite")
        eval_prompts = prompt_gen.get_evaluation_prompts()

        metric = GEval(
            model=eval_llm,
            task_introduction=eval_prompts['task_introduction'],
            evaluation_criteria=eval_prompts['evaluation_criteria']
        )

        geval_input = f"""
        Question: {question}
        Expected Answer: {expected_answer}
        Model Answer: {model_output}
        """
        
        score_result = metric.score(output=geval_input)
        
        if score_result is None:
            raise ValueError("GEval metric.score() returned None.")

        return {
            'geval_score': score_result.value,
            'geval_reason': score_result.reason 
        }
    except litellm.RateLimitError as e:
        print(f"\n\033[91mRate limit hit. Returning 0 for GEval. Error: {e}\033[0m")
        return {'geval_score': 0.0, 'geval_reason': f"Rate limit error: {e}"}
    except Exception as e:
        print(f"\nError calculating GEval: {e}")
        return {'geval_score': 0.0, 'geval_reason': str(e)}
    
def load_model_answers(run_folder, model_id, question_file_prefix):
    model_path = os.path.join(run_folder, model_id)
    all_items = []
    for fname in sorted(os.listdir(model_path)):
        if fname.startswith(question_file_prefix) and fname.endswith(".json"):
            with open(os.path.join(model_path, fname), "r") as f:
                data = json.load(f)
                all_items.append((fname, data[1:]))
    return all_items

def group_by_question(data):
    grouped = defaultdict(list)
    for _, items in data:
        for entry in items:
            grouped[entry["question"]].append(entry)
    return grouped

def select_run(settings):
    base_folder = settings.experiments_folder
    print (f"Available runs in \033[1m{base_folder}\033[0m:")
    available_runs = sorted([
        d for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ])
    if not available_runs:
        print("No runs found.")
        return None, None
    print("Available runs:")
    for idx, run in enumerate(available_runs, 1):
        print(f"{idx}. {run}")
    run_choice = input("Enter a run number (default [1]): ").strip()
    run_idx = int(run_choice) - 1 if run_choice.isdigit() and 1 <= int(run_choice) <= len(available_runs) else 0
    selected_run = available_runs[run_idx]
    run_path = os.path.join(base_folder, selected_run)
    return selected_run, run_path

def select_model(selected_run, run_path):
    available_models = sorted([
        d for d in os.listdir(run_path)
        if os.path.isdir(os.path.join(run_path, d))
    ])
    if not available_models:
        print("No models found in this run.")
        return []

    print(f"\nAvailable models in \033[1m{selected_run}\033[0m:")
    for idx, model in enumerate(available_models, 1):
        print(f"{idx}. {model}")
        
    model_choice = input("Enter model number or 'A' to run all models (default: all): ").strip().lower()

    if model_choice == 'a' or model_choice == '':
        print("\nSelecting all available models for evaluation.")
        return available_models
    elif model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
        selected_model_id = available_models[int(model_choice) - 1]
        print(f"\nSelected model: \033[92m{selected_model_id}\033[0m")
        return [selected_model_id]
    else:
        print("Invalid selection. Aborting.")
        return []

def main():
    settings = Settings()
    
    prompt_gen = PromptGenerator(settings=settings)

    selected_run, run_path = select_run(settings)
    if not selected_run:
        return

    selected_models_ids = select_model(selected_run, run_path)
    if not selected_models_ids:
        return
    
    for model_id in selected_models_ids:
        model_path = os.path.join(run_path, model_id)

        raw_data = load_model_answers(run_path, model_id, settings.files['question_file_name'])
        grouped = group_by_question(raw_data)

        print(f"\n=== Evaluating model responses for {model_id} and logging to Opik ===")

        results = []
        for q_idx, (question, answers) in enumerate(grouped.items(), 1):
            print(f"\n{q_idx:02d}. \033[1mQuestion:\033[0m {question.strip()}")
            expected_answer = answers[0]["correct_answer"]
            print(f"     \033[96mExpected Answer:\033[0m {expected_answer.strip()}")

            trace_metadata = {
                "model_id": model_id,
                "run_name": selected_run,
                "question_text": question.strip()
            }

            trace = opik_client.trace(
                name=f"eval_{model_id}_q{q_idx}",
                metadata=trace_metadata
            )

            required_phrases_for_contains = [word.strip() for word in expected_answer.split() if word.strip()]
            if len(required_phrases_for_contains) > 5:
                required_phrases_for_contains = required_phrases_for_contains[:5]

            for run_idx, run in enumerate(answers[:settings.num_runs_per_question], 1):
                model_output = run["raw_answer"]

                rouge_scores = calculate_rouge(model_output, expected_answer)
                contains_results = check_contains(model_output, required_phrases_for_contains)
                geval_scores = calculate_geval(model_output, expected_answer, question, prompt_gen)

                span_output = {**rouge_scores, **contains_results, **geval_scores}

                span = trace.span(
                    name=f"run_{run_idx}",
                    input={
                        "model_output": model_output,
                        "expected_answer": expected_answer
                    },
                    output={model_output},
                    metadata={"run_iteration": run_idx}
                )
                
                scores_to_log = []

                # Add ROUGE and Contains
                all_scores = {**rouge_scores, **contains_results}

                for name, value in all_scores.items():
                    scores_to_log.append({
                        "id": span.id,
                        "name": name,
                        "value": float(value)
                    })

                # GEval score
                scores_to_log.append({
                    "id": span.id,
                    "name": "geval_score",
                    "value": geval_scores['geval_score'],
                    "reason": geval_scores.get('geval_reason', '')
                })

                # Log the scores to the span
                opik_client.log_spans_feedback_scores(scores=scores_to_log)

                print(f"   Run {run_idx}:")
                print(f"     \033[93mModel Answer:\033[0m {model_output.strip()}")
                print("     --- ROUGE Scores ---")
                for metric_name, score_value in rouge_scores.items():
                    print(f"         \033[92m{metric_name.replace('_', ' ').title()}:\033[0m {score_value:.4f}")
                print("     --- Contains Results ---")
                if "contains_percentage" in contains_results:
                    print(f"         \033[92mContains Percentage:\033[0m {contains_results['contains_percentage']:.2f}%")
                if "all_contained" in contains_results:
                    print(f"         \033[92mAll Contained:\033[0m {bool(contains_results['all_contained'])}")
                print("     --- GEval Score ---")
                print(f"         \033[92mGEval Score:\033[0m {geval_scores['geval_score']:.1f}")
                print(f"         \033[92mGEval Reason:\033[0m {geval_scores['geval_reason']}")


                results.append({
                    "question": question,
                    "run": run_idx,
                    "model_output": model_output,
                    "expected_answer": expected_answer,
                    "rouge_scores": rouge_scores,
                    "contains_results": contains_results,
                    "geval_scores": geval_scores,
                })
            
            trace.end()
            

        output_path = os.path.join(model_path, settings.evaluation_output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to: {output_path}")
    print(f"\n\033[1mEvaluation results available on Opik/Comet project: 'LLMmark_evaluation'\033[0m")

if __name__ == "__main__":
    main()