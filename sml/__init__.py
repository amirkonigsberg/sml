"""Self-Monitored Learning (SML).

A reference specification for the architecture proposed in:

    Self-Monitored Learning: An Architectural Proposal for Endogenous
    Epistemic Feedback in Language Models. Konigsberg, A. (2026).

This package defines the architectural components of SML at the level of
class definitions, method signatures, and reference implementations of the
operations that admit clean specification without a training pipeline.
"""

from .types import (
    EpistemicConfig,
    TrainingConfig,
    RetrievalResult,
)
from .memory import EpistemicMemory
from .encoder import ContextEncoder, select_for_memory
from .epistemic_module import EpistemicModule, compute_confidence_signal
from .losses import calibration_loss, consistency_loss, total_loss
from .inference import sml_forward, generate, MainModelProtocol

__all__ = [
    # Configuration
    "EpistemicConfig",
    "TrainingConfig",
    "RetrievalResult",
    # Components
    "EpistemicMemory",
    "ContextEncoder",
    "EpistemicModule",
    "MainModelProtocol",
    # Operations
    "compute_confidence_signal",
    "select_for_memory",
    # Losses
    "calibration_loss",
    "consistency_loss",
    "total_loss",
    # Inference
    "sml_forward",
    "generate",
]

__version__ = "0.1.0"
