import requests
from openai import OpenAI
import time

class OpenAIClient:
    def __init__(self, api_key: str,  base_url: str, max_requests_per_minute=0):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_requests_per_minute = max_requests_per_minute

    def chat(self, model, messages, stream=False):
        response = self.client.chat.completions.create(model=model, messages=messages)
        if "gemini" in model:
            # Take into account Google Generative Language API quota limits
            # This affects the mearured response time
            if self.max_requests_per_minute > 0:
                time.sleep(60 / self.max_requests_per_minute)
            return response.choices[0].message.content
        if "gpt" in model:
            return response.output_text
        return ""
