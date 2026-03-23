from openai import OpenAI

from apps.core.config import get_settings
from apps.core.llm.base import BaseLLM

settings = get_settings()


class VLLMLLM(BaseLLM):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model

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
