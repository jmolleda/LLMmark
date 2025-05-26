import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models-api')))

from ollama import Client   
from settings import Settings

def main():
    settings = Settings("../models-api/config.yaml")

    client = Client(host=settings.ollama['host'])

    response_chat = client.chat("gemma3:1b", [{"role": "user", "content": "Why is the sky blue?"}])
    print("Chat response:", response_chat['message']['content'])

    response_generate = client.generate("gemma3:1b", "Why is the sky blue?")
    print("Generate response:", response_generate['response'])

if __name__ == "__main__":
    main()