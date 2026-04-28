"""Shared type definitions for the SML architecture.

This module defines the named tensor types and dataclasses used across the
architectural components. The types are intentionally explicit so that the
shapes of the tensors flowing through the system are visible from the
function signatures alone.

The conventions follow the paper's notation: T is the sequence length,
B is the batch size, D is the main model's hidden dimension, d is the
epistemic state dimension, d_k is the key dimension, d_v is the value
dimension, and K is the number of nearest neighbors retrieved from the
epistemic memory.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor


# Named tensor type aliases. These are not enforced at runtime but make the
# shapes of the tensors flowing through the system visible from signatures.

HiddenStates = Tensor  # (B, T, D), the main model's hidden states
TokenLogits = Tensor   # (B, T, V), where V is vocabulary size
EpistemicState = Tensor  # (B, T, d), the epistemic state vector e_t
MemoryKey = Tensor       # (N, d_k), where N is the number of memory entries
MemoryValue = Tensor     # (N, d_v)
Query = Tensor           # (B, T, d_k)
RetrievedValues = Tensor  # (B, T, K, d_v)
ConfidenceSignal = Tensor  # (B, T), entropy of the main model's distribution


@dataclass
class EpistemicConfig:
    """Configuration for the epistemic module and memory.

    Attributes:
        d: Dimensionality of the epistemic state vector. Default 64 for
            small models, 256 for frontier-scale models.
        d_k: Dimensionality of the memory keys. Should match the output of
            the encoder's key projection.
        d_v: Dimensionality of the memory values. Should match the output of
            the encoder's value projection.
        K: Number of nearest neighbors retrieved from the memory at each
            generation step.
        memory_size: Maximum number of entries in the epistemic memory. Older
            or less informative entries are pruned when the memory exceeds
            this size.
        retrieval_temperature: Temperature for the softmax over retrieval
            scores. Default 1.0.
        beta: Coefficient relating the readout to the temperature in the
            consistency loss. See losses.py.
    """

    d: int = 64
    d_k: int = 128
    d_v: int = 128
    K: int = 8
    memory_size: int = 1_000_000
    retrieval_temperature: float = 1.0
    beta: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for the multi-task loss weights and training schedule.

    Attributes:
        lambda_cal: Weight on the calibration loss in the multi-task objective.
        lambda_cons: Weight on the consistency loss in the multi-task
            objective.
        memory_update_frequency: How often (in training steps) the memory is
            re-encoded to reflect the model's evolving error patterns.
        memory_inclusion_threshold: Loss threshold above which a context is
            included in the epistemic memory.
    """

    lambda_cal: float = 0.1
    lambda_cons: float = 0.1
    memory_update_frequency: int = 1000
    memory_inclusion_threshold: float = 2.0


class RetrievalResult(NamedTuple):
    """Result of querying the epistemic memory.

    Fields:
        values: The retrieved values, aggregated by attention over the top-K
            neighbors. Shape (B, T, d_v).
        attention_weights: The attention weights over the top-K neighbors.
            Shape (B, T, K). Useful for analysis and ablation.
        retrieved_indices: The indices into the memory of the retrieved
            entries. Shape (B, T, K). Useful for tracing which memory entries
            contributed to a given prediction.
    """

    values: Tensor
    attention_weights: Tensor
    retrieved_indices: Tensor
