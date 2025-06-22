import ollama
# from opik import track, opik_context


class OllamaClient:
    def __init__(self, host):
        self.client = ollama.Client(host)

    # @track(project_name="LLMmark_clients", tags=['ollama', 'python-library'])
    def chat(self, model, messages, stream=False, temperature=0.7):
        
        options = {
            'temperature': temperature,
            'top_p': 0.1,
            'seed': 27
        }
        
        response = self.client.chat(model=model, messages=messages, stream=stream, options=options)
        return response['message']['content']

    def generate(self, model, prompt, temperature=0.7):
        options = {
            'temperature': temperature
        }
        response = self.client.generate(model=model, prompt=prompt, options=options)
        return response['response']