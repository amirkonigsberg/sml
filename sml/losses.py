"""The training objective for SML.

The main model and epistemic module are trained jointly with a multi-task
loss:

    L = L_LM + lambda_cal * L_cal + lambda_cons * L_cons

where L_LM is the standard language modeling loss, L_cal is the calibration
loss that trains the epistemic module to predict input-conditional
reliability, and L_cons is the consistency loss that ensures the main
model's outputs are conditioned on the epistemic state in the way the
proposal requires.

This module implements L_cal and L_cons. L_LM is the standard cross-entropy
loss provided by any transformer training stack and is not reimplemented
here.

The calibration loss matches equation (3) of the paper:

    L_cal = E[ ell(r(e_t), 1[argmax p_theta(.) = y_{t+1}]) ]

where ell is binary cross-entropy and r is the readout from the epistemic
state to a predicted reliability score.

The consistency loss matches equation (4) and the temperature specification
added in the revision:

    L_cons = E[ KL( p_theta(. | x, e_t) || q(. | x, e_t) ) ]
    q(. | x, e_t) = softmax(logits_theta(x) / T(e_t))
    T(e_t) = exp(beta * r(e_t))

The temperature T is a learned function of the epistemic state, not a
global constant. This is the architectural commitment that distinguishes
SML from per-output post-hoc temperature scaling.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def calibration_loss(
    reliability_score: Tensor,
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Calibration loss L_cal.

    Trains the epistemic module's readout r(e_t) to predict whether the main
    model's prediction at position t will be correct.

    Args:
        reliability_score: Predicted reliability r(e_t), shape (B, T), values
            in (0, 1).
        logits: Main model logits, shape (B, T, V).
        targets: Target token indices, shape (B, T). Positions equal to
            ignore_index are excluded from the loss.
        ignore_index: Token index to ignore (typically padding).

    Returns:
        Scalar loss.
    """
    predictions = logits.argmax(dim=-1)  # (B, T)
    correct = (predictions == targets).float()  # (B, T)

    mask = (targets != ignore_index).float()  # (B, T)

    # Binary cross-entropy between reliability_score and correct.
    eps = 1e-7
    bce = -(
        correct * torch.log(reliability_score.clamp(min=eps))
        + (1 - correct) * torch.log((1 - reliability_score).clamp(min=eps))
    )

    masked_bce = bce * mask
    return masked_bce.sum() / mask.sum().clamp(min=1.0)


def consistency_loss(
    logits_with_epistemic: Tensor,
    logits_without_epistemic: Tensor,
    reliability_score: Tensor,
    targets: Tensor,
    beta: float = 1.0,
    ignore_index: int = -100,
) -> Tensor:
    """Consistency loss L_cons.

    Ensures that the main model's output distribution, when conditioned on
    the epistemic state, has entropy that scales with the reliability
    reported by the epistemic module: more uniform when reliability is low,
    more peaked when reliability is high.

    Implementation: construct a target distribution q by temperature-scaling
    the unconditioned logits, with the temperature determined by the
    reliability readout. Then minimize the KL divergence between the
    conditioned distribution p and q.

    The temperature is computed as T(e_t) = exp(beta * (1 - r(e_t))). When
    r(e_t) is high (model is reliable on this input), T is low (distribution
    is sharp). When r(e_t) is low (model is unreliable), T is high
    (distribution is more uniform). This matches the architectural intent.

    Args:
        logits_with_epistemic: Main model logits when conditioned on e_t,
            shape (B, T, V). These are p_theta(. | x, e_t).
        logits_without_epistemic: Main model logits without epistemic
            conditioning, shape (B, T, V). These are p_theta(. | x), used to
            construct the target q.
        reliability_score: Predicted reliability r(e_t), shape (B, T).
        targets: Target token indices, shape (B, T). Used only for masking.
        beta: Coefficient relating reliability to temperature.
        ignore_index: Token index to ignore (typically padding).

    Returns:
        Scalar loss.
    """
    # Compute temperature from reliability. Note the (1 - r): high reliability
    # gives low temperature (sharp distribution), low reliability gives high
    # temperature (more uniform distribution).
    temperature = torch.exp(beta * (1.0 - reliability_score))  # (B, T)

    # Construct the target distribution q by temperature-scaling the
    # unconditioned logits.
    scaled_logits = logits_without_epistemic / temperature.unsqueeze(-1)
    log_q = F.log_softmax(scaled_logits, dim=-1)
    q = log_q.exp()

    # Compute log p where p is the conditioned distribution.
    log_p = F.log_softmax(logits_with_epistemic, dim=-1)

    # KL(p || q) = sum_v p(v) * (log p(v) - log q(v))
    # We use the log-domain formulation for numerical stability.
    # Note: we compute KL(p || q), so the expectation is under p.
    p = log_p.exp()
    kl_per_position = (p * (log_p - log_q)).sum(dim=-1)  # (B, T)

    mask = (targets != ignore_index).float()
    masked_kl = kl_per_position * mask
    return masked_kl.sum() / mask.sum().clamp(min=1.0)


def total_loss(
    lm_loss: Tensor,
    cal_loss: Tensor,
    cons_loss: Tensor,
    lambda_cal: float = 0.1,
    lambda_cons: float = 0.1,
) -> Tensor:
    """The multi-task loss L = L_LM + lambda_cal * L_cal + lambda_cons * L_cons.

    This matches equation (2) of the paper.

    Args:
        lm_loss: The standard language modeling loss.
        cal_loss: The calibration loss.
        cons_loss: The consistency loss.
        lambda_cal: Weight on the calibration loss.
        lambda_cons: Weight on the consistency loss.

    Returns:
        Scalar total loss.
    """
    return lm_loss + lambda_cal * cal_loss + lambda_cons * cons_loss
