import time
import json

from ..clients.openai_client import OpenAIClient


class LLMJudge:
    def __init__(self, client: OpenAIClient, model="models/gemini-1.5-flash"):
        self.client = client
        self.model = model
        self.history = []

    def generate_prompt(self, question, expected_answer, model_answer):
        prompt = f"""You are a strict but fair exam evaluator.

        Question:
        {question}

        Expected answer:
        {expected_answer}

        Answer provided by the model:
        {model_answer}

        Evaluate the quality of the model's answer based on:
        - Correctness
        - Relevance
        - Clarity

        Provide your output as a JSON object containing:
        - "grade": a float score between 0.0 (very poor) and 1.0 (perfect)
        - "justification": a short justification explaining the score

        Example:
        {{"grade": 0.85, "justification": "The answer is mostly correct, with minor omissions."}}"""
        
        return prompt

    def eval(self, question, expected_answer, model_answer):
        prompt = self.generate_prompt(question, expected_answer, model_answer)
        messages = [{"role": "user", "content": prompt}]
        start_time = time.time()
        response = self.client.chat(model=self.model, messages=messages)

        latency = round(time.time() - start_time, 3)

        raw_text = response.strip()

        # Clean up markdown-style code block if present
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("` \n")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print("Failed to parse model output as JSON.", e)
            result = {
                "grade": None,
                "justification": f"Invalid response format:\n{raw_text}"
            }

        result["latency"] = latency
        self.history.append(result)
        return result