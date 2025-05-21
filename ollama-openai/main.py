from ollama_openai import OllamaOpenAIClient, ChatRunner

if __name__ == "__main__":
    stream_input = input("Do you want to run in streaming mode? (Y/n): ").strip().lower()
    stream = (stream_input == "" or stream_input == "y")
    client = OllamaOpenAIClient()
    runner = ChatRunner(client)
    runner.run(
        model="gemma3:1b",
        prompt="What are the main differences between LLaMA 2 and LLaMA 3?",
        stream=stream
    )