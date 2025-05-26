# Ollamark
Ollamark is a simple tool for solving examns interacting with large language models (LLMs).
Ollama local models are supported.

## Usage Example

See `examples/example.py` for a minimal example.

## Multiple-choice question

question.txt

What is the representation of the number 61 in natural binary? 
[a] 00111110000
[b] 00111101
[c] 0011110110
[d] 43h
[e] 10101110
[f] 101001001

<b> # Correct answer

## Output

output.json

[
  {
    "Num. correct answers": 0,
    "Accuracy": 0.0,
    "Avg_response_time (s.)": 0.09
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