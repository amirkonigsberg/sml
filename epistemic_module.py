"""The epistemic module E_phi.

The epistemic module takes three inputs: a representation of the main
model's hidden states at the current position, a memory-conditioned
representation produced by querying the epistemic memory, and an explicit
confidence signal derived from the main model's output distribution. It
produces an epistemic state vector e_t that is fed back into the main model
to condition its next-token distribution.

This is the architectural core of SML. The forward pass implements
equation (1) of the paper:

    e_t = E_phi(h_t, m_t, c_t)

The module is implemented as a small MLP in this specification. A full
implementation might use a small transformer for richer interaction between
the three inputs, but the MLP is sufficient to demonstrate the architectural
proposal.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .memory import EpistemicMemory
from .types import (
    ConfidenceSignal,
    EpistemicConfig,
    EpistemicState,
    HiddenStates,
    Query,
)


class EpistemicModule(nn.Module):
    """The epistemic module E_phi.

    Produces the epistemic state vector e_t given the main model's hidden
    states, the epistemic memory, and the main model's confidence signal.

    The module owns the query projection that maps hidden states to memory
    queries, the aggregation network that combines the three input streams,
    and the readout head that maps the epistemic state to a predicted
    reliability score (used by the calibration loss).
    """

    def __init__(
        self,
        hidden_dim: int,
        config: EpistemicConfig,
        memory: EpistemicMemory,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.memory = memory

        # Query projection: maps main model hidden states to memory queries.
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, config.d_k),
            nn.LayerNorm(config.d_k),
        )

        # Aggregation network: combines hidden states, retrieved values, and
        # confidence signal into the epistemic state vector.
        # The input dimension is hidden_dim + d_v + 1 (the confidence signal
        # is a scalar per position).
        self.aggregate = nn.Sequential(
            nn.Linear(hidden_dim + config.d_v + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d),
            nn.LayerNorm(config.d),
        )

        # Readout head: maps the epistemic state to a predicted reliability
        # score in (0, 1). Used by the calibration loss to train the
        # epistemic state to track input-conditional reliability.
        self.readout = nn.Sequential(
            nn.Linear(config.d, config.d // 2),
            nn.GELU(),
            nn.Linear(config.d // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: HiddenStates,
        confidence_signal: ConfidenceSignal,
    ) -> tuple[EpistemicState, Tensor]:
        """Compute the epistemic state vector for each position.

        Args:
            hidden_states: Main model hidden states, shape (B, T, D).
            confidence_signal: Per-position confidence signal derived from
                the main model's output distribution. Shape (B, T). The
                default specification uses the entropy of the next-token
                distribution; richer specifications are possible.

        Returns:
            A tuple (epistemic_state, reliability_score) where
            - epistemic_state has shape (B, T, d) and is the vector e_t fed
              back into the main model.
            - reliability_score has shape (B, T) and is the readout r(e_t)
              used by the calibration loss.
        """
        # Compute queries from the main model's hidden states.
        queries = self.query_proj(hidden_states)  # (B, T, d_k)

        # Query the epistemic memory.
        retrieved = self.memory.query(queries)
        memory_conditioned = retrieved.values  # (B, T, d_v)

        # Combine the three inputs.
        combined = torch.cat(
            [
                hidden_states,
                memory_conditioned,
                confidence_signal.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Produce the epistemic state.
        epistemic_state = self.aggregate(combined)  # (B, T, d)

        # Compute the reliability readout.
        reliability_score = self.readout(epistemic_state).squeeze(-1)  # (B, T)

        return epistemic_state, reliability_score


def compute_confidence_signal(
    logits: Tensor,
    method: str = "entropy",
) -> ConfidenceSignal:
    """Compute a confidence signal from the main model's output distribution.

    Args:
        logits: Main model logits, shape (B, T, V).
        method: Which confidence signal to compute. Currently supported:
            "entropy": The entropy of the next-token distribution. Higher
                entropy means lower confidence.
            "max_prob": The maximum probability assigned by the distribution.
                Higher max probability means higher confidence.

    Returns:
        A confidence signal of shape (B, T).
    """
    if method == "entropy":
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
    elif method == "max_prob":
        probs = torch.softmax(logits, dim=-1)
        return probs.max(dim=-1).values
    else:
        raise ValueError(f"Unknown confidence signal method: {method}")
