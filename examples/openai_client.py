import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models-api')))

from openai import OpenAI
from rich import print as rprint
from openai_client import OpenAIClient

def test_openai_gpt():
    rprint("[green]Testing GPT[/green]")

    # Will rise ERROR 429 if no credit is available
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1/"
        )

    try:
        response = client.chat(
            model="gpt-4.1",
            messages="Write a one-sentence bedtime story about a unicorn."
        )
        print(response)
    except Exception as e:
        # Expected error if no credit is available
        rprint(f"An error occurred: {e}")
        # return

def test_openai_gemini():
    rprint("[green]Testing Gemini[/green]")
    client = OpenAIClient(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )    

    response = client.chat("gemini-2.0-flash", 
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Why is the sky blue?"
            }
        ]
    )
    
    print("Response from Gemini:")
    print(response)    

def main():
    test_openai_gpt()
    test_openai_gemini()

if __name__ == "__main__":
    main()
