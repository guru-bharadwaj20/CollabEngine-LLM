"""Model-serving backends.

One interface, several implementations. Swapping between them is a config line,
so the harness developed against `MockBackend` on a laptop runs unchanged against
vLLM on a 24 GB card.

`MockBackend` is not only a stand-in for absent hardware: it is the testbed that
validates the causal machinery. It can be configured to produce genuinely
specialized agents, undifferentiated agents, or purely positional behavior, which
lets the ablation and analysis stack be checked against known ground truth before
a single GPU token is spent.
"""

from collabengine.backends.base import (
    ChatMessage,
    GenRequest,
    GenResponse,
    LLMBackend,
)
from collabengine.backends.mock import MockBackend, MockMode

__all__ = [
    "ChatMessage",
    "GenRequest",
    "GenResponse",
    "LLMBackend",
    "MockBackend",
    "MockMode",
]
