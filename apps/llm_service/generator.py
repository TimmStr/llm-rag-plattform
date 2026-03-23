from apps.core.llm.base import BaseLLM


class Generator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, query: str, contexts: list[str]):
        context_str = "\n\n".join(contexts)

        prompt = f"""
Answer the question based on the context.

Context:
{context_str}

Question:
{query}
"""
        return self.llm.generate(prompt)
