# Architecture Notes

This document complements the paper by providing additional notes on the architectural choices, the design space around them, and the considerations that shape how the components in this repository fit together. It is not a substitute for the paper. Read the paper first, particularly section 4.

## The forward pass

A self-monitored forward pass at position t proceeds as follows.

1. The main model produces hidden states `h_t` and unconditioned logits `logits_uncond` from the input sequence.
2. A confidence signal `c_t` is computed from `logits_uncond`. The default specification uses entropy of the next-token distribution.
3. The epistemic module takes `h_t`, queries the epistemic memory using a learned query projection, aggregates the retrieved values by attention, combines all three input streams, and produces the epistemic state `e_t`.
4. The main model is run again, this time with `e_t` provided as additional conditioning, to produce the final logits `logits_cond`.

Step 4 is where the architecture commits to a particular integration mechanism. The default in this repository is concatenation to the input embedding: the epistemic state is projected into the embedding space and concatenated to the token embedding before the main model's first layer. Alternatives include adding to a designated layer's residual stream and providing the epistemic state as the key-value source for a cross-attention operation. These are documented as design choices to be evaluated empirically.

## The epistemic memory

The memory is a key-value store. Keys live in `d_k`-dimensional space and values live in `d_v`-dimensional space. Both are learned by the encoder, which is trained jointly with the rest of the system through gradient flow from the calibration and consistency losses.

The memory is constructed during training. When a training example produces a prediction with high loss or poor calibration, the encoder converts the context to a key-value pair and the pair is added to the memory. The threshold above which a context is included is a hyperparameter (`memory_inclusion_threshold` in `TrainingConfig`).

The retrieval operation, given a query `k_q`, returns the top-K nearest neighbors by cosine similarity, then aggregates their values by softmax-weighted attention. The temperature of the softmax (`retrieval_temperature` in `EpistemicConfig`) controls how sharply attention concentrates on the nearest neighbor. At temperature 1.0, the operation is standard scaled attention. At lower temperatures, retrieval becomes more like a hard nearest-neighbor lookup. At higher temperatures, retrieval becomes more like uniform aggregation over the top-K.

The memory is bounded in size. When it exceeds `memory_size`, entries are pruned. The default pruning policy in this specification is FIFO (oldest entries removed first). A full implementation would maintain retention scores based on:

- Recency: how recently the entry was added.
- Calibration impact: whether the entry's inclusion measurably improves calibration on a held-out set.
- Retrieval frequency: how often the entry has been retrieved by recent queries.

These retention signals would be combined into a score, and the lowest-scoring entries would be pruned. The pruning policy is a research direction in itself; this specification flags it without committing.

## The training objective

The multi-task loss is

```
L = L_LM + lambda_cal * L_cal + lambda_cons * L_cons
```

`L_LM` is the standard language modeling loss. `L_cal` trains the epistemic module's readout to predict whether the main model's prediction at each position will be correct. `L_cons` trains the main model's conditioned distribution to match a target distribution constructed by temperature-scaling the unconditioned logits, with the temperature determined by the reliability readout.

The temperature is `T(e_t) = exp(beta * (1 - r(e_t)))`. Note the `1 - r`: when reliability is high, `1 - r` is low, so the exponent is small, so the temperature is close to 1.0 (sharp distribution). When reliability is low, `1 - r` is high, so the temperature is high, so the distribution becomes more uniform. This is the architectural intent: when the model is reliable on this input, let it be confident; when the model is unreliable, force it to be less confident.

`beta` controls how strongly reliability translates into temperature changes. At `beta = 0`, reliability has no effect and the consistency loss reduces to KL between the conditioned and unconditioned distributions, which is uninformative. At very high `beta`, low reliability collapses the distribution to uniform, which is too aggressive. Reasonable starting values are `beta` in the range 0.5 to 2.0.

## The choice of confidence signal

The default confidence signal is the entropy of the next-token distribution. Alternatives include:

- Maximum probability (max-margin between the most likely token and the rest).
- Energy: the negative log-sum-exp of the logits, which measures the overall scale of the distribution and has been shown to be useful for OOD detection.
- A learned confidence head on top of the main model's hidden states.

The entropy choice is made for simplicity and because entropy is the most direct measure of uncertainty in a distribution. A full implementation would likely make the confidence signal a configuration option.

## Ablations the architecture invites

The architecture makes a series of choices, each of which can be ablated to test its contribution. The natural ablation set:

1. Remove the memory and use only `h_t` and `c_t` as inputs to the epistemic module. Tests whether the memory is doing meaningful work.
2. Remove the consistency loss. Tests whether conditioning on the epistemic state improves calibration without explicit pressure.
3. Remove the calibration loss. Tests whether the epistemic state can produce useful conditioning without an explicit reliability target.
4. Replace the learned encoder with a simple pooled hidden state. Tests whether the encoder's projections add information beyond what is already in the hidden states.
5. Replace softmax-attention retrieval with hard top-1 nearest neighbor. Tests whether the soft aggregation matters.

These are the kinds of experiments that a full implementation would run. The ablation set is offered here as a checklist for anyone extending the specification.

## Computational cost

At inference time, SML adds:

- The cost of the query projection, which is `O(D * d_k)` per position.
- The cost of the memory lookup, which is `O(N * d_k)` per position where N is the memory size, dominated by the similarity computation. Approximate nearest-neighbor methods (HNSW, IVF) reduce this to roughly `O(d_k * log N)`.
- The cost of the epistemic module's forward pass, which is the cost of a small MLP and is negligible compared to the main model's forward pass.
- The cost of running the main model twice (once unconditioned, once conditioned). This is the largest single cost. It can be amortized by running both passes within a single optimized pass through the main model with conditional layers, but the simple specification doubles the inference cost.

For frontier-scale models, the doubled forward pass dominates. We expect SML inference to be 1.5x to 2x slower than baseline inference in a naive implementation, with the gap shrinking to 1.1x to 1.3x in an optimized implementation.

At training time, the additional costs are:

- The encoder's forward and backward pass on memory-bound contexts.
- The memory storage cost, which scales with the memory size.
- The double main-model pass, as above.

Training cost is therefore meaningfully higher than baseline training. A full implementation would need to evaluate whether the calibration and reliability gains justify the cost.

## Open questions

This specification leaves several design questions open. They are flagged for anyone extending the implementation:

1. **How to integrate `e_t` into the main model.** Concatenation to embeddings is the default. Cross-attention and residual addition are alternatives. The choice may interact with model scale.

2. **How to construct the encoder's training signal.** The encoder is trained implicitly through gradient flow from the calibration and consistency losses. A more direct objective (for example, contrastive learning over keys for similar-reliability contexts) might produce sharper memory entries.

3. **How to handle distribution shift in the memory.** The memory is constructed from the training distribution. When the deployed model encounters out-of-distribution inputs, the retrieved memory entries may be uninformative. A version of SML with online memory updates is a natural extension but raises questions about when updates are safe.

4. **How the epistemic module and main model interact during long generations.** The current specification queries the memory at each generation step. Whether this is necessary, or whether epistemic state can be carried forward across steps with periodic memory queries, is an empirical question.

5. **Whether the architecture generalizes to multimodal and RL settings.** The paper sketches the generalizations but the specification here is for autoregressive language models only.
