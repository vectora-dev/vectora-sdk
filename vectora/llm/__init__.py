from vectora.exceptions import ComingSoonError


def __getattr__(name: str) -> None:
    raise ComingSoonError(
        "vectora.llm is coming soon. LLM observability ships in a later Vectora release."
    )
