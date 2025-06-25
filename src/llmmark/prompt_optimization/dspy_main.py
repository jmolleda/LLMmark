import dspy
import logging
import requests
import json

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
noisy_loggers = ["httpx", "urllib3", "asyncio", "comet_ml", "dspy", "litellm"]
for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

OLLAMA_HOST = 'http://localhost:11434'
OLLAMA_MODEL_ID = 'deepseek-r1:1.5b'
MAX_TOKENS = 4096

def is_ollama_model_installed(model_name: str, ollama_base_url: str) -> bool:
    """Check if a model is installed in Ollama."""
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status() # Raise an exception for HTTP errors
        models = response.json().get("models", [])
        return any(model.get("name", "").startswith(model_name) for model in models)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking Ollama model list at {ollama_base_url}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while checking Ollama models: {e}")
        return False

try:
    if not is_ollama_model_installed(OLLAMA_MODEL_ID, OLLAMA_HOST):
        logging.error(f"Ollama model '{OLLAMA_MODEL_ID}' not found or Ollama server not running at {OLLAMA_HOST}. Please pull the model (ollama pull {OLLAMA_MODEL_ID}) or start the server (ollama serve).")
        exit(1)

    local_lm = dspy.LM(
        f'ollama_chat/{OLLAMA_MODEL_ID}',
        api_base=OLLAMA_HOST,
        api_key='',
        max_tokens=MAX_TOKENS
    )
    logging.info(f"DSPy configured with local Ollama model: {local_lm.model}")

    dspy.settings.configure(lm=local_lm)

except Exception as e:
    logging.error(f"Failed to configure DSPy with local Ollama. Error: {e}", exc_info=True)
    exit(1)

question_example = {
    "question": "What is the representation of 61 in hexadecimal?",
    "answer": "3D",
}

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers, including a thought process."""
    question = dspy.InputField(desc="The question to answer")
    thought: str = dspy.OutputField(desc="Your thinking process (step-by-step reasoning).")
    answer: str = dspy.OutputField(desc="The final answer (often between 1 and 5 words).")


class CustomChainOfThought(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(BasicQA)

    def forward(self, question):
        full_prompt = f"""
        Answer questions with short factoid answers.
        Think step-by-step before answering.
        Your thought process must be clear and detailed, but concise.
        DO NOT repeat steps or get stuck in a loop.
        Ensure your reasoning concludes clearly before the final answer.
        Your final answer must be concise.

        Question: {question}

        Respond with a JSON object containing 'thought' and 'answer' fields.
        Example: {{"thought": "First, I'll identify...", "answer": "..."}}
        """

        prediction = self.predict(question=question, __instruction=full_prompt)
        
        return prediction


generate_answer = CustomChainOfThought()

logging.info(f"\nProcessing Question: {question_example['question']}")
try:
    pred = generate_answer(question=question_example['question'])

    print(f"\nQuestion: {question_example['question']}")
    print(f"Thought: {pred.thought}")
    print(f"Answer: {pred.answer}")

except Exception as e:
    logging.error(f"An error occurred during prediction: {e}", exc_info=True)
    logging.info("\n--- Manual parsing fallback (if Predict still fails) ---")
    if dspy.settings.lm.history:
        raw_output_entry = dspy.settings.lm.history[-1]
        raw_model_response_content = None
        if isinstance(raw_output_entry, dict) and 'response' in raw_output_entry:
            response_data = raw_output_entry['response']
            if isinstance(response_data, dict) and 'choices' in response_data:
                raw_model_response_content = response_data['choices'][0]['message']['content']
            elif hasattr(response_data, 'choices') and len(response_data.choices) > 0 and hasattr(response_data.choices[0], 'message'):
                raw_model_response_content = response_data.choices[0].message.content
        
        if raw_model_response_content:
            print(f"\nRaw Model's Full Response (from history):\n{raw_model_response_content}")
            try:
                parsed_json = json.loads(raw_model_response_content)
                print(f"\nManually Parsed JSON: {parsed_json}")
                print(f"Manually Extracted Thought (from JSON): {parsed_json.get('thought', 'N/A')}")
                print(f"Manually Extracted Answer (from JSON): {parsed_json.get('answer', 'N/A')}")
            except json.JSONDecodeError:
                print("Raw response is not valid JSON.")
            except Exception as parse_e:
                print(f"Error during manual JSON parsing: {parse_e}")
        else:
            print("No raw model response content in history for manual parsing.")
    else:
        print("LM history is empty for manual parsing.")


print("\n--- Full LM Call History (Local Model) ---")
dspy.settings.lm.inspect_history()