import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"


class LLMService:
    def __init__(self, model="llama3.2:3b-instruct-q4_0"):
        self.model = model

    def stream(self, prompt: str):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]