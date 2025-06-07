import requests
from openai import OpenAI, InternalServerError
import time

from opik.integrations.openai import track_openai


class OpenAIClient:
    def __init__(self, api_key: str,  base_url: str, max_requests_per_minute=0):
        self.client = track_openai(OpenAI(api_key=api_key, base_url=base_url))
        self.max_requests_per_minute = max_requests_per_minute

    def chat(self, model, messages, stream=False):
        response = self.client.chat.completions.create(model=model, messages=messages)
        if "gemini" in model:
            retries = 3
            for attempt in range(retries):
                try:
                    # Take into account Google Generative Language API quota limits
                    # This affects the mearured response time
                    if self.max_requests_per_minute > 0:
                        time.sleep(60 / self.max_requests_per_minute)
                    return response.choices[0].message.content
                except InternalServerError as e:
                    if e.status_code == 503:
                        print(f"\033[93m[503 Model Overload] Retrying in 15 seconds...(Attempt {retries}/{self.max_retries})\033[0m")
                        time.sleep(15)
                    else:
                        print(f"\033[91m[Error] {e}\033[0m")
                        break

        if "gpt" in model:
            return response.output_text
        return ""
