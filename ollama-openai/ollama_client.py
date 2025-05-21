# This script demonstrates how to use the Ollama API with OpenAI's Python client.
# Make sure to install the required packages:
import ollama
import openai

class OpenAIClient:
    """
    OpenAIClient provides an interface compatible with OpenAI's API for interacting with Ollama models.

    Attributes:
        client: An instance of openai.OpenAI configured to communicate with the specified base_url and api_key.

    Args:
        base_url (str): The base URL for the Ollama API. Defaults to "http://localhost:11434/v1".
        api_key (str): The API key for authentication. Defaults to "ollama".

    Methods:
        chat(model, messages, stream=True):
            Sends a chat completion request to the Ollama API using the specified model and messages.
            Args:
                model (str): The name of the model to use for chat completion.
                messages (list): A list of message dictionaries following the OpenAI chat format.
                stream (bool): Whether to stream responses. Defaults to True.
            Returns:
                The response from the Ollama API, either as a stream or a complete result.
    """
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
