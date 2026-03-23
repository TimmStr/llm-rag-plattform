import os

from openai import OpenAI

from apps.core.llm.base import BaseLLM
from dotenv import load_dotenv


def load_openai_key() -> str:
    load_dotenv()
    return os.getenv("OPENAI_KEY")


class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=load_openai_key())
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
