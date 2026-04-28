"""The epistemic memory.

The epistemic memory M is a set of key-value pairs (k_i, v_i) where the keys
are learned compressed representations of input contexts and the values are
learned representations of the prediction-failure history associated with
those contexts. The memory is constructed during training and queried during
both training and inference.

This module specifies the memory's interface: how entries are added, how the
memory is queried, and how aggregation works. The retrieval operation is the
one piece of the architecture that admits a clean reference implementation
without training infrastructure, so it is implemented in full. The memory's
construction during training is a stub, since it requires a training loop
that is out of scope for this specification repository.

The retrieval operation matches equation (5) in the paper:

    alpha_j = softmax(<k_q, k_{i_j}> / tau)
    m_t = sum_j alpha_j * v_{i_j}

where the softmax is taken over the top-K nearest neighbors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import (
    EpistemicConfig,
    MemoryKey,
    MemoryValue,
    Query,
    RetrievalResult,
)


class EpistemicMemory(nn.Module):
    """The epistemic memory M.

    Stores key-value pairs representing the model's prediction-failure history.
    Supports adding new entries, querying by similarity to a query vector, and
    pruning to maintain a bounded size.

    The memory is implemented as a pair of tensors (keys and values) plus a
    metadata structure tracking insertion order and retention scores. In a
    full implementation, the keys and values would be stored on accelerator
    memory or, for very large memories, on disk with an approximate nearest-
    neighbor index. This specification uses simple tensors; scaling
    considerations are documented in docs/ARCHITECTURE.md.
    """

    def __init__(self, config: EpistemicConfig) -> None:
        super().__init__()
        self.config = config

        # The memory is initialized empty. Entries are added during training.
        # We register the memory as a buffer rather than a parameter, since
        # the memory contents are not directly optimized by gradient descent;
        # they are constructed from the encoder's outputs on training examples
        # where the model's prediction was incorrect or poorly calibrated.
        self.register_buffer(
            "keys",
            torch.zeros(0, config.d_k),
            persistent=True,
        )
        self.register_buffer(
            "values",
            torch.zeros(0, config.d_v),
            persistent=True,
        )

    def __len__(self) -> int:
        """Return the current number of entries in the memory."""
        return self.keys.shape[0]

    def add(self, keys: MemoryKey, values: MemoryValue) -> None:
        """Add new entries to the memory.

        Args:
            keys: New keys to add, shape (n, d_k).
            values: New values to add, shape (n, d_v).

        After adding, if the memory exceeds config.memory_size, the least
        informative entries are pruned. The pruning policy is a stub; in a
        full implementation, retention scores based on recency, calibration
        impact, and retrieval frequency would determine which entries to
        remove.
        """
        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                f"keys and values must have the same number of rows; "
                f"got {keys.shape[0]} and {values.shape[0]}"
            )
        if keys.shape[1] != self.config.d_k:
            raise ValueError(
                f"keys must have d_k={self.config.d_k} columns; "
                f"got {keys.shape[1]}"
            )
        if values.shape[1] != self.config.d_v:
            raise ValueError(
                f"values must have d_v={self.config.d_v} columns; "
                f"got {values.shape[1]}"
            )

        self.keys = torch.cat([self.keys, keys], dim=0)
        self.values = torch.cat([self.values, values], dim=0)

        if len(self) > self.config.memory_size:
            self._prune()

    def _prune(self) -> None:
        """Prune the memory to config.memory_size entries.

        This is a stub. A full implementation would maintain retention scores
        (based on recency, calibration impact, retrieval frequency) and
        remove the lowest-scoring entries. The default policy here is FIFO:
        the oldest entries are removed first.
        """
        excess = len(self) - self.config.memory_size
        if excess > 0:
            self.keys = self.keys[excess:]
            self.values = self.values[excess:]

    def query(self, queries: Query) -> RetrievalResult:
        """Query the memory by similarity.

        Implements the retrieval operation specified in equation (5) of the
        paper. For each query vector, retrieve the top-K nearest neighbors by
        cosine similarity, then aggregate their values by softmax-weighted
        attention.

        Args:
            queries: Query vectors, shape (B, T, d_k).

        Returns:
            A RetrievalResult containing:
            - values: Aggregated retrieved values, shape (B, T, d_v).
            - attention_weights: Attention weights over the top-K neighbors,
              shape (B, T, K).
            - retrieved_indices: Indices into the memory, shape (B, T, K).
        """
        if len(self) == 0:
            # If the memory is empty, return zeros. This is the case at the
            # start of training, before any failures have been encoded.
            B, T, _ = queries.shape
            return RetrievalResult(
                values=torch.zeros(
                    B, T, self.config.d_v, device=queries.device
                ),
                attention_weights=torch.zeros(
                    B, T, self.config.K, device=queries.device
                ),
                retrieved_indices=torch.zeros(
                    B, T, self.config.K, dtype=torch.long, device=queries.device
                ),
            )

        K = min(self.config.K, len(self))
        B, T, d_k = queries.shape

        # Cosine similarity between each query and each memory key.
        # Shape: (B*T, N) where N is the number of memory entries.
        flat_queries = queries.reshape(B * T, d_k)
        normed_queries = torch.nn.functional.normalize(flat_queries, dim=-1)
        normed_keys = torch.nn.functional.normalize(self.keys, dim=-1)
        scores = normed_queries @ normed_keys.t()

        # Top-K nearest neighbors.
        topk_scores, topk_indices = scores.topk(K, dim=-1)

        # Softmax over the top-K to produce attention weights.
        # The temperature controls how sharply attention concentrates on the
        # nearest neighbor.
        attention_weights = torch.softmax(
            topk_scores / self.config.retrieval_temperature, dim=-1
        )

        # Gather the values corresponding to the retrieved indices.
        retrieved = self.values[topk_indices]  # (B*T, K, d_v)

        # Aggregate by attention.
        aggregated = (attention_weights.unsqueeze(-1) * retrieved).sum(dim=-2)

        return RetrievalResult(
            values=aggregated.reshape(B, T, self.config.d_v),
            attention_weights=attention_weights.reshape(B, T, K),
            retrieved_indices=topk_indices.reshape(B, T, K),
        )
