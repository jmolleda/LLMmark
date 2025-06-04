import time

from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key: str,  base_url: str, max_requests_per_minute=0):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_requests_per_minute = max_requests_per_minute

    def chat(self, model, messages, stream=False):
        response = self.client.chat.completions.create(model=model, messages=messages)

        # Take into account Google Generative Language API quota limits for Gemini models
        if "gemini" in model and self.max_requests_per_minute > 0:
            # This affects the measured response time
            time.sleep(60 / self.max_requests_per_minute)

        # The OpenAI Python client returns the response text in the same way for
        # both GPT and Gemini models
        return response.choices[0].message.content
