from ollama import chat
from pydantic import BaseModel
import json


class Answer(BaseModel):
  answer: str

  json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "grade": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "A numerical score representing the quality of reasoning, where 0.0 is very poor and 1.0 is perfect."
            },
            "justification": {
                "type": "string",
                "description": "A brief justification of the grade given, explaining the reasoning quality and any relevant details."
            }
        },
        "required": ["grade", "justification"]
    }
  )




def get_formatted_answer(answer: str) -> chr:

  reasoning_trace = answer

  response = chat(
  messages=[
    {
        "role": "system",
        "content": f"You are a helpful assistant that evaluates reasoning quality and translates it into JSON according to this schema: {Answer.model_json_schema()}"
    },
    {
      'role': 'user',
      'content': reasoning_trace,
    }
  ],
  model='Osmosis/Osmosis-Structure-0.6B',
  format=Answer.model_json_schema(),
  )

  answer = Answer.model_validate_json(response.message.content)
  print(answer)