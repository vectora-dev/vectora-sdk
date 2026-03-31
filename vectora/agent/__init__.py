from vectora.exceptions import ComingSoonError


def __getattr__(name: str) -> None:
    raise ComingSoonError(
        "vectora.agent is coming soon. Agent tracing ships in a later Vectora release."
    )
