import ollama

class OllamaClient:
    def __init__(self, host):
        self.client = ollama.Client(host)

    def chat(self, model, messages, stream=False):
        return self.client.chat(model=model, messages=messages, stream=stream)
    
    def generate(self, model, prompt):
        return self.client.generate(model=model,prompt=prompt)