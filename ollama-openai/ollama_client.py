# This script demonstrates how to use the Ollama API with OpenAI's Python client.
# Make sure to install the required packages:
import ollama
import openai

class OpenAIClient:
    def __init__(self, base_url="http://localhost:11434/v1", api_key="ollama"):
        # OpenAI-compatible client for Ollama
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def chat(self, model, messages, stream=True):
        # Use Ollama's native Python client for chat
        return ollama.chat(
            model=model,
            messages=messages,
            stream=stream
        )

class ChatRunner:
    def __init__(self, client):
        self.client = client

    def run(self, model, prompt, stream=True):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(model=model, messages=messages, stream=stream)
        if stream:
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
            print()
        else:
            print(response['message']['content'])
