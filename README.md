# Self-Monitored Learning (SML)

A reference specification for the architecture proposed in:

**Self-Monitored Learning: An Architectural Proposal for Endogenous Epistemic Feedback in Language Models**
Amir Konigsberg, 2026.
[Link to paper / arXiv preprint forthcoming]

## What this repository is

This is a specification repository, not a working implementation. The code here defines the architectural components of Self-Monitored Learning (SML) at the level of class definitions, method signatures, and reference implementations of the operations that are simple enough to write in code without a training pipeline. The repository is intended to be read alongside the paper. Each file corresponds to a component of the architecture specified in section 4.

The purpose of the repository is to make the architectural proposal precise enough that a reader can see exactly what is being proposed, and to provide a starting point for anyone who wants to extend the specification into a working implementation.

## What this repository is not

This repository does not contain a working SML system. There is no training code, no learned weights, no benchmarks, and no claims about empirical performance. The class skeletons run as Python and pass type checks, but they do not produce trained models. Anyone seeking to test SML empirically will need to build out the training infrastructure, the data loading, the optimization, and the evaluation harness. This is the kind of contribution we are inviting; see `CONTRIBUTING.md`.

## What SML is

SML is a class of learning architectures in which a language model maintains internal representations of its own uncertainty, its own prediction-failure history on related inputs, and its own confidence calibration as first-class features of its forward pass. The architectural mechanism that enables this, which we call endogenous epistemic feedback, closes a feedback loop that current learning paradigms leave open.

A self-monitored autoregressive language model consists of:

- A main model `M_theta`, a standard transformer.
- An epistemic module `E_phi`, a smaller network that takes the main model's hidden states, queries an epistemic memory, and produces an epistemic state vector.
- An epistemic memory `M`, a key-value store of compressed prediction-failure history constructed during training.
- A joint training objective that includes a language modeling loss, a calibration loss, and a consistency loss.

At inference, the main model's next-token distribution is conditioned on the epistemic state vector produced by the epistemic module. This is the architectural sense in which the loop is closed: the model's relationship to its own outputs is part of how it produces them.

For the full specification, see the paper.

## Repository structure

```
sml/
  epistemic_module.py    The epistemic module E_phi
  memory.py              The epistemic memory M and its operations
  encoder.py             The encoder f_enc that constructs memory entries
  losses.py              Calibration loss and consistency loss
  inference.py           The inference-time forward pass with epistemic conditioning
  types.py               Shared type definitions and named tensors

tests/
  test_shapes.py         Unit tests verifying type signatures and tensor shapes

docs/
  ARCHITECTURE.md        Detailed architectural notes that complement the paper
  GLOSSARY.md            Definitions of key terms
```

## Reading order

For someone coming to this repository for the first time, we suggest:

1. Read the paper, particularly section 4.
2. Read `docs/ARCHITECTURE.md` for the architectural notes.
3. Read `sml/types.py` to see the shared type definitions.
4. Read `sml/memory.py` and `sml/epistemic_module.py` for the two main components.
5. Read `sml/losses.py` for the training objective.
6. Read `sml/inference.py` for how the components compose at inference.

## Installation

This repository is a specification. The Python code is intentionally minimal and depends only on PyTorch:

```bash
pip install -e .
```

## Citation

If you build on this specification, please cite the paper:

```
@article{konigsberg2026sml,
  title={Self-Monitored Learning: An Architectural Proposal for Endogenous Epistemic Feedback in Language Models},
  author={Konigsberg, Amir},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT. See `LICENSE`.

## Contact

For questions, comments, or collaboration: amirkonigsberg@gmail.com
