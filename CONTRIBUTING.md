# Contributing

This is a specification repository, not a working implementation. The intent is to make the architectural proposal in the paper precise enough that someone reading the paper and the code together can see exactly what is being proposed.

The kinds of contributions that would be most valuable, in rough priority order:

## Extending the specification into a working implementation

The single most valuable contribution would be a working implementation of SML on a small open model (GPT-2 small, Pythia-160M, or similar). This means:

- Adapting an existing transformer to satisfy the `MainModelProtocol` interface, including a mechanism to condition the forward pass on an epistemic state vector.
- Building the training loop that maintains the epistemic memory across batches.
- Running the joint training of main model, epistemic module, and encoder on a standard pretraining or fine-tuning dataset.
- Implementing the inference loop with KV cache integration.

This is roughly two to four weeks of focused work for someone with the right background. Anyone interested in this contribution should reach out before starting; we want to coordinate to avoid duplicated effort.

## Empirical evaluation

Once a working implementation exists, the architecture invites a series of evaluations:

- Calibration on standard benchmarks (MMLU, MMLU-Pro, BIG-Bench).
- Hallucination rates on factual question-answering with proximity-controlled prompts.
- Behavior under distribution shift, comparing in-distribution and OOD calibration.
- Sycophancy resistance using the standard sycophancy probes.

Comparisons should be against a matched baseline (the same main model trained with the same data and compute, without the epistemic module or memory).

## Ablations

The architecture makes a series of choices that should be ablated:

- Memory removed (epistemic module receives only `h_t` and `c_t`).
- Consistency loss removed.
- Calibration loss removed.
- Hard top-1 retrieval instead of soft attention.
- Different confidence signals (entropy, max-prob, energy).
- Different integration mechanisms for the epistemic state (concatenation, cross-attention, residual addition).

A systematic ablation study would tell us which architectural commitments are doing real work and which are arbitrary.

## Architectural variants

Several variants of SML are flagged in the paper and in `docs/ARCHITECTURE.md` as future work:

- A parametric memory variant in which the prediction-failure history is distilled into the epistemic module's weights rather than stored in an explicit memory.
- An adversarial training variant in which the epistemic module is trained to find the main model's failure modes.
- An integrated architecture in which epistemic information is carried by designated dimensions of the main model's residual stream rather than by a separate module.
- Multimodal and RL extensions.

Each of these is a legitimate research direction.

## Documentation and clarification

The specification is intentionally minimal. Documentation contributions are welcome:

- Clarifications of the architectural notes in `docs/ARCHITECTURE.md`.
- Worked examples showing how to integrate SML with existing transformer libraries.
- Notes on practical considerations (memory size at scale, retrieval latency, training stability).

## What we are not looking for

- Speculative extensions that drift far from the paper's argument.
- Pull requests that add features without first discussing whether they fit the project.
- Cosmetic changes that don't clarify the architecture.

## Code style

Python 3.10+. Type hints on all public functions. Docstrings explaining what each component does and how it relates to the paper. Tests for any new component, at minimum verifying tensor shapes.

## Contact

Open an issue or email amirkonigsberg@gmail.com.
