# Ollamark
Ollamark is a simple tool for solving examns interacting with large language models (LLMs).
Ollama local models are supported.

## Usage Example

See `examples/ollama_client.py` for a minimal example using Ollama local models.
See `examples/openai_client.py` for a minimal example using the OpenAI API to access online models.

## Multiple-choice question file

question.txt

```
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

```
{
    "num_experiments": 10,
    "num_correct": 8,
    "num_incorrect": 2,
    "accuracy": 0.8,
    "average_response_time": 0.47,
    "total_response_time (s.)": 4.729
}
```

run_1/gemini-2.0-flash/question.json

```
[
    {
        "num_correct": 0,
        "accuracy": 0.0,
        "average_response_time (s)": 0.09
    },
    {
        "question": "What is the representation of the number 61 in natural binary? \n[a] 00111110000\n[b] 00111101\n[c] 0011110110\n[d] 43h\n[e] 10101110\n[f] 101001001",
        "answer": "[a]",
        "correct_answer": "[b]",
        "raw_answer": "[a]",
        "model": "Gemma3:1b",
        "response_time (s.)": 0.091
    }
]
```