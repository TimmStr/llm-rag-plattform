from openai import OpenAI

from apps.core.config import get_settings
from apps.core.llm.base import BaseLLM

settings = get_settings()


class OpenAILLM(BaseLLM):
    def __init__(self, model: str):
        self.client = OpenAI(api_key=settings.openai_token)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    print(OpenAILLM().generate("Wie heißt die Hauptstadt von Deutschland?"))
