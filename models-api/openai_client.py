import requests
from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key: str,  base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model, messages, stream=False):
        response = self.client.chat.completions.create(model=model, messages=messages)
        if "gemini" in model:
            return response.choices[0].message.content
        if "gpt" in model:
            return response.output_text
        return ""
