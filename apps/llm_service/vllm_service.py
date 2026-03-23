import os

from dotenv import load_dotenv
from openai import OpenAI

from apps.core.llm.base import BaseLLM


def vllm_config() -> dict[str, str]:
    load_dotenv()
    return {
        "url": os.getenv("VLLM_URL"),
        "api_key": os.getenv("VLLM_KEY")
    }


class VLLMLLM(BaseLLM):
    def __init__(self):
        conn_config = vllm_config()
        base_url: str = conn_config["url"]
        self.client = OpenAI(
            base_url=base_url,
            api_key=conn_config["api_key"]
        )
        self.model = "Qwen/Qwen3-0.6B"

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
