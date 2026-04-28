"""The encoder f_enc.

The encoder takes a context in which the model's prediction was incorrect or
poorly calibrated, and produces a key-value pair to be added to the epistemic
memory. The keys are trained to be similar for contexts with similar
reliability profiles, and the values compress the information needed by the
epistemic module to estimate input-conditional reliability.

The encoder is trained jointly with the main model and the epistemic module.
Its training objective is implicit: the encoder is updated through the
gradients of the calibration and consistency losses, since the keys it
produces determine which memory entries are retrieved at inference time and
the values it produces determine what information the epistemic module has
access to. There is no separate loss on the encoder itself.

The encoder is implemented as a small neural network that takes a
representation of the context (typically a pooled representation of the
main model's hidden states over the input window) and produces a key and a
value. The architecture is intentionally minimal in this specification; a
full implementation would likely use a small transformer or MLP, depending
on the scale of the main model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import EpistemicConfig, HiddenStates, MemoryKey, MemoryValue


class ContextEncoder(nn.Module):
    """The encoder f_enc that constructs memory entries.

    Takes a representation of an input context and produces a key-value pair
    suitable for insertion into the epistemic memory.

    The context representation is expected to be a pooled summary of the
    main model's hidden states over the input window where the prediction
    was made. Pooling strategies (mean, last-token, learned attention pool)
    are a design choice; the default here is mean pooling.
    """

    def __init__(
        self,
        hidden_dim: int,
        config: EpistemicConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        # Key projection: maps from main model hidden states to the key space.
        # Trained to produce keys similar for contexts with similar
        # reliability profiles.
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim, config.d_k),
            nn.LayerNorm(config.d_k),
        )

        # Value projection: maps from main model hidden states to the value
        # space. Trained to produce values that, when aggregated, give the
        # epistemic module useful information about the model's reliability
        # on the current input.
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_dim, config.d_v),
            nn.LayerNorm(config.d_v),
        )

    def forward(
        self,
        hidden_states: HiddenStates,
        attention_mask: Tensor | None = None,
    ) -> tuple[MemoryKey, MemoryValue]:
        """Encode a batch of contexts into key-value pairs.

        Args:
            hidden_states: Main model hidden states, shape (B, T, D).
            attention_mask: Optional attention mask, shape (B, T). Positions
                where the mask is zero are excluded from pooling.

        Returns:
            A tuple (keys, values) where keys has shape (B, d_k) and values
            has shape (B, d_v). One key-value pair is produced per context.
        """
        pooled = self._pool(hidden_states, attention_mask)
        keys = self.key_proj(pooled)
        values = self.value_proj(pooled)
        return keys, values

    def _pool(
        self,
        hidden_states: HiddenStates,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Pool hidden states across the sequence dimension.

        Default: mean pooling, masking out padding positions if a mask is
        provided. A full implementation would likely make the pooling
        strategy a configuration option.
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked.sum(dim=1) / denom


def select_for_memory(
    losses: Tensor,
    threshold: float,
) -> Tensor:
    """Select which contexts in a batch should be added to the epistemic memory.

    The default policy: include contexts where the per-example loss exceeds
    a threshold. Other policies (calibration error, retrieval-based novelty)
    are possible and are documented in docs/ARCHITECTURE.md.

    Args:
        losses: Per-example losses, shape (B,).
        threshold: Loss threshold above which a context is included.

    Returns:
        A boolean mask of shape (B,) indicating which contexts to include.
    """
    return losses > threshold
