# LLMmark
LLMmark interacts with large language models (LLMs) to solve exams.

Local models are supported using [Ollama](https://github.com/ollama/ollama).

Online models are supported via the [OpenAI API Python library](https://github.com/openai/openai-python).

## API keys
This project uses API keys from environment variables.

```bash
export GEMINI_API_KEY=<YOUR_API_KEY_HERE>
source ~/.bashrc
```

## Usage Example

See `examples/ollama_client.py` for a minimal example using Ollama local models.

See `examples/openai_client.py` for a minimal example using the OpenAI API to access online models.

## Multiple-choice question file

question.txt

```txt
What is the representation of the number 61 in natural binary?
[a] 00111110000  
[b] 00111101  
[c] 0011110110  
[d] 43h  
[e] 10101110  
[f] 101001001  

<b> 
```

## Output files

run_1/gemini-2.0-flash/stats.json

```json
{
  "num_experiments": 10,
  "num_correct": 9,
  "num_incorrect": 1,
  "accuracy": 0.9,
  "average_response_time (s)": 0.622,
  "total_response_time (s)": 6.221
}
```

run_1/gemini-2.0-flash/question.json

```json
[
  {
    "num_correct": 1,
    "accuracy": 1.0,
    "averaga_response_time (s)": 0.524
  },
  {
    "question": "What is the representation of the number 61 in natural binary? \n[a] 00111110000\n[b] 00111101\n[c] 0011110110\n[d] 43h\n[e] 10101110\n[f] 101001001\n",
    "answer": "[b]",
    "correct_answer": "[b]",
    "raw_answer": "[b]",
    "model": "Gemini 2.0 Flash",
    "response_time (s)": 0.524
  }
]
```