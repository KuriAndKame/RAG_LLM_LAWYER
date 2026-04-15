import requests
import json
from typing import List, Dict, Optional
from config.settings import settings


class LLMClient:
    def __init__(self):
        self.base_url = settings.LLM_BASE_URL.rstrip(
            '/')  # http://localhost:1234
        self.model = settings.LLM_MODEL_NAME

    def generate(self,
                 messages: List[Dict[str, str]],
                 temperature: float = None,
                 max_tokens: int = None) -> str:
        # Используем OpenAI-совместимый путь
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}

        print(f"Sending to {url}:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Response body: {response.text}")
            raise Exception(f"LM Studio request failed: {e}")
        except KeyError:
            raise Exception(f"Unexpected response format: {data}")
