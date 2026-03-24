from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


__all__ = ["GenerationOutput", "LLM", "SamplingParams", "__version__"]


try:
    __version__ = version("inferlib")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str):
    if name == "LLM":
        from inferlib.api import LLM

        return LLM
    if name == "SamplingParams":
        from inferlib.types import SamplingParams

        return SamplingParams
    if name == "GenerationOutput":
        from inferlib.types import GenerationOutput

        return GenerationOutput
    raise AttributeError(name)
