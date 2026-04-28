"""Unit tests verifying that the architectural components have the expected
type signatures and produce tensors of the expected shapes.

These are not tests of correctness in the empirical sense; the components
are not trained and produce arbitrary output. The tests verify only that
the architecture is internally consistent at the level of shapes and types.
This is sufficient for a specification repository: if a component has the
wrong shape signature, the architectural proposal has a bug.
"""

from __future__ import annotations

import torch

from sml import (
    ContextEncoder,
    EpistemicConfig,
    EpistemicMemory,
    EpistemicModule,
    calibration_loss,
    compute_confidence_signal,
    consistency_loss,
    total_loss,
)


def test_memory_query_returns_correct_shapes() -> None:
    config = EpistemicConfig(d=64, d_k=128, d_v=128, K=4)
    memory = EpistemicMemory(config)

    # Add some entries.
    n_entries = 20
    keys = torch.randn(n_entries, config.d_k)
    values = torch.randn(n_entries, config.d_v)
    memory.add(keys, values)

    assert len(memory) == n_entries

    # Query the memory.
    B, T = 2, 5
    queries = torch.randn(B, T, config.d_k)
    result = memory.query(queries)

    assert result.values.shape == (B, T, config.d_v)
    assert result.attention_weights.shape == (B, T, config.K)
    assert result.retrieved_indices.shape == (B, T, config.K)


def test_memory_query_empty_returns_zeros() -> None:
    config = EpistemicConfig(d=64, d_k=128, d_v=128, K=4)
    memory = EpistemicMemory(config)

    B, T = 2, 5
    queries = torch.randn(B, T, config.d_k)
    result = memory.query(queries)

    assert result.values.shape == (B, T, config.d_v)
    assert torch.all(result.values == 0.0)


def test_encoder_produces_correct_shapes() -> None:
    hidden_dim = 256
    config = EpistemicConfig(d=64, d_k=128, d_v=128)
    encoder = ContextEncoder(hidden_dim, config)

    B, T = 4, 10
    hidden_states = torch.randn(B, T, hidden_dim)

    keys, values = encoder(hidden_states)

    assert keys.shape == (B, config.d_k)
    assert values.shape == (B, config.d_v)


def test_epistemic_module_produces_correct_shapes() -> None:
    hidden_dim = 256
    config = EpistemicConfig(d=64, d_k=128, d_v=128, K=4)
    memory = EpistemicMemory(config)

    # Add entries so the memory query returns non-zero results.
    memory.add(torch.randn(50, config.d_k), torch.randn(50, config.d_v))

    module = EpistemicModule(hidden_dim, config, memory)

    B, T = 2, 8
    hidden_states = torch.randn(B, T, hidden_dim)
    confidence = torch.randn(B, T)

    epistemic_state, reliability = module(hidden_states, confidence)

    assert epistemic_state.shape == (B, T, config.d)
    assert reliability.shape == (B, T)
    assert torch.all(reliability >= 0.0)
    assert torch.all(reliability <= 1.0)


def test_confidence_signal_entropy() -> None:
    B, T, V = 2, 5, 100
    logits = torch.randn(B, T, V)

    entropy = compute_confidence_signal(logits, method="entropy")
    assert entropy.shape == (B, T)
    # Entropy is non-negative.
    assert torch.all(entropy >= 0.0)


def test_confidence_signal_max_prob() -> None:
    B, T, V = 2, 5, 100
    logits = torch.randn(B, T, V)

    max_prob = compute_confidence_signal(logits, method="max_prob")
    assert max_prob.shape == (B, T)
    assert torch.all(max_prob >= 0.0)
    assert torch.all(max_prob <= 1.0)


def test_calibration_loss() -> None:
    B, T, V = 2, 5, 100
    reliability = torch.rand(B, T)
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    loss = calibration_loss(reliability, logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_consistency_loss() -> None:
    B, T, V = 2, 5, 100
    logits_with = torch.randn(B, T, V)
    logits_without = torch.randn(B, T, V)
    reliability = torch.rand(B, T)
    targets = torch.randint(0, V, (B, T))

    loss = consistency_loss(logits_with, logits_without, reliability, targets)
    assert loss.dim() == 0
    # KL divergence is non-negative.
    assert loss.item() >= -1e-5  # numerical tolerance


def test_total_loss_combines_components() -> None:
    lm = torch.tensor(2.0)
    cal = torch.tensor(0.5)
    cons = torch.tensor(0.3)

    total = total_loss(lm, cal, cons, lambda_cal=0.1, lambda_cons=0.2)
    expected = 2.0 + 0.1 * 0.5 + 0.2 * 0.3
    assert abs(total.item() - expected) < 1e-6


if __name__ == "__main__":
    test_memory_query_returns_correct_shapes()
    test_memory_query_empty_returns_zeros()
    test_encoder_produces_correct_shapes()
    test_epistemic_module_produces_correct_shapes()
    test_confidence_signal_entropy()
    test_confidence_signal_max_prob()
    test_calibration_loss()
    test_consistency_loss()
    test_total_loss_combines_components()
    print("All tests passed.")
