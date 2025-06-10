import ollama
# from opik import track, opik_context


class OllamaClient:
    def __init__(self, host):
        self.client = ollama.Client(host)

    # @track(project_name="LLMmark_clients", tags=['ollama', 'python-library'])
    def chat(self, model, messages, stream=False):
        response = self.client.chat(model=model, messages=messages, stream=stream)
        return response['message']['content']
    
    def generate(self, model, prompt):
        response = self.client.generate(model=model, prompt=prompt)
        return response['response']