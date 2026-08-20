"""Studio: process / data / model management (Dify + LangChain + DataRobot analog).

Python still never calls a model provider directly. LLM steps in a process use
deterministic local providers (echo / template / trained classifier) or can be
wired to an existing evalloop task eval. Trained models are in-process AutoML
artifacts, not promptfoo providers.
"""

from evalloop.studio.errors import StudioError

__all__ = ["StudioError"]
