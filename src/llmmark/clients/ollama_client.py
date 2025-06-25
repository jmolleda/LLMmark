import ollama
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, host, top_p=0.1, seed=27, timeout=120):
        try:
            self.client = ollama.Client(host=host, timeout=timeout)
            self.top_p = top_p
            self.seed = seed
            self.client.list()
            logger.info(
                f"Ollama client initialized for host: {host} with a {timeout}s timeout."
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to Ollama at {host}. Please ensure Ollama is running. Error: {e}"
            )
            exit(1)

    def chat(self, model, messages, stream=False, temperature=0.0):
        options = {"temperature": temperature, "top_p": self.top_p, "seed": self.seed, "num_predict": 4096}
        try:
            response = self.client.chat(
                model=model, messages=messages, stream=stream, options=options
            )
            return response["message"]["content"]
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API Error for model {model}: {e.error} (Status: {e.status_code})"
            )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred with Ollama client for model {model}: {e}"
            )
            return None

    def generate(self, model, prompt, temperature=0.0):
        options = {"temperature": temperature, "top_p": self.top_p, "seed": self.seed}
        try:
            response = self.client.generate(model=model, prompt=prompt, options=options)
            return response["response"]
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API Error for model {model}: {e.error} (Status: {e.status_code})"
            )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred with Ollama client for model {model}: {e}"
            )
            return None