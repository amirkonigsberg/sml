"""The inference-time forward pass for SML.

This module defines the protocol that a self-monitored language model
follows at inference time. It is implemented as a callable that takes a
main model, an epistemic module, and an input sequence, and produces the
next-token distribution conditioned on the epistemic state.

The protocol matches the inference-time procedure specified in section 4
of the paper:

    1. Run the main model's forward pass to produce hidden states h_t.
    2. Compute the confidence signal c_t from the main model's logits.
    3. Run the epistemic module to produce the epistemic state e_t,
       which involves querying the epistemic memory.
    4. Run the main model's forward pass again, this time conditioned on
       e_t, to produce the final logits.
    5. Sample the next token from these logits.

Step 4 requires the main model to accept the epistemic state as additional
conditioning. The exact mechanism is a design choice (concatenation to the
input embedding, addition to a residual stream, cross-attention). The
default specification here is concatenation to the input embedding.

The inference loop here is sketched as pseudocode in the docstring of the
generate function. A full implementation would integrate with the main
model's KV cache and would handle the additional cost of querying the
epistemic memory at each generation step.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .epistemic_module import EpistemicModule, compute_confidence_signal
from .types import EpistemicState, HiddenStates, TokenLogits


class MainModelProtocol(Protocol):
    """The interface a main model must satisfy to be used with SML.

    The main model is expected to be a transformer that can be run in two
    modes: standard (without epistemic conditioning) and epistemic-conditioned
    (with an epistemic state vector provided).

    A full implementation would adapt an existing transformer (e.g., a
    HuggingFace model) to satisfy this interface, typically by adding a
    projection from the epistemic state space into the input embedding
    space and concatenating or adding the projected state to the embeddings.
    """

    def forward_unconditioned(
        self,
        input_ids: Tensor,
    ) -> tuple[HiddenStates, TokenLogits]:
        """Forward pass without epistemic conditioning."""
        ...

    def forward_conditioned(
        self,
        input_ids: Tensor,
        epistemic_state: EpistemicState,
    ) -> tuple[HiddenStates, TokenLogits]:
        """Forward pass with epistemic conditioning."""
        ...


def sml_forward(
    main_model: MainModelProtocol,
    epistemic_module: EpistemicModule,
    input_ids: Tensor,
) -> tuple[TokenLogits, EpistemicState, Tensor]:
    """Run the SML forward pass.

    Pseudocode (matching Algorithm 1 in the paper):

        1. h, logits_uncond <- main_model.forward_unconditioned(input_ids)
        2. c <- compute_confidence_signal(logits_uncond)
        3. e, r <- epistemic_module(h, c)
        4. _, logits_cond <- main_model.forward_conditioned(input_ids, e)
        5. return logits_cond, e, r

    Args:
        main_model: The main model M_theta. Must satisfy MainModelProtocol.
        epistemic_module: The epistemic module E_phi.
        input_ids: Input token indices, shape (B, T).

    Returns:
        A tuple (logits, epistemic_state, reliability_score):
        - logits: The final next-token logits conditioned on the epistemic
          state, shape (B, T, V).
        - epistemic_state: The epistemic state e_t at each position, shape
          (B, T, d).
        - reliability_score: The readout r(e_t) at each position, shape
          (B, T).
    """
    hidden_states, logits_uncond = main_model.forward_unconditioned(input_ids)

    confidence = compute_confidence_signal(logits_uncond, method="entropy")

    epistemic_state, reliability_score = epistemic_module(
        hidden_states, confidence
    )

    _, logits_cond = main_model.forward_conditioned(input_ids, epistemic_state)

    return logits_cond, epistemic_state, reliability_score


def generate(
    main_model: MainModelProtocol,
    epistemic_module: EpistemicModule,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> Tensor:
    """Autoregressive generation with SML.

    Generates max_new_tokens tokens by repeatedly running the SML forward
    pass and sampling from the resulting distribution.

    This is a sketch. A full implementation would use the main model's KV
    cache to avoid recomputing hidden states for the prefix at each step,
    and would batch the epistemic memory query across positions where
    possible.

    Args:
        main_model: The main model.
        epistemic_module: The epistemic module.
        prompt_ids: The prompt, shape (B, T_prompt).
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature applied to the final logits.

    Returns:
        The generated token sequence including the prompt, shape
        (B, T_prompt + max_new_tokens).
    """
    sequence = prompt_ids
    for _ in range(max_new_tokens):
        logits, _, _ = sml_forward(main_model, epistemic_module, sequence)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat([sequence, next_token], dim=1)
    return sequence
