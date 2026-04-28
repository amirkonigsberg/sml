# Glossary

Definitions of the key terms used in the paper and this repository.

**Self-Monitored Learning (SML).** A class of learning architectures in which a model maintains internal representations of its own uncertainty, its own prediction-failure history on related inputs, and its own confidence calibration as first-class features of its forward pass. The headline term for the proposal.

**Endogenous Epistemic Feedback (EEF).** The architectural mechanism that makes a system self-monitored. A learning paradigm exhibits endogenous epistemic feedback when the model's relationship to its own outputs (its uncertainty, its history of being wrong, its calibration) is represented inside the model and accessible to it during generation, rather than being computed about the model from outside.

**Open-loop architecture.** A learning architecture in which feedback (loss, reward, gradient) acts on the model from outside and leaves no trace inside it that the model can query during inference. Most current learning paradigms (supervised, RLHF, constitutional AI, process reward models) are open-loop in this sense.

**Closed-loop architecture.** A learning architecture in which the model's relationship to its own outputs is part of how it produces them. SML is a proposal for a closed-loop architecture for autoregressive language models.

**Epistemic state.** The vector `e_t` produced by the epistemic module at each position. It is a compact representation of the system's reliability assessment for the current input, available to the main model during generation.

**Epistemic module.** The network `E_phi` that takes the main model's hidden states, queries the epistemic memory, and produces the epistemic state vector. Trained jointly with the main model.

**Epistemic memory.** The key-value store `M` that holds compressed prediction-failure history. Constructed during training from contexts where the model's prediction was incorrect or poorly calibrated.

**Reliability score.** The scalar `r(e_t)` produced by the readout from the epistemic state. Trained by the calibration loss to predict whether the main model's prediction at the current position will be correct.

**Calibration loss.** The loss `L_cal` that trains the epistemic module's readout to predict input-conditional reliability.

**Consistency loss.** The loss `L_cons` that trains the main model's conditioned output distribution to have entropy that scales with the reliability reported by the epistemic module.

**Encoder.** The network `f_enc` that converts a context where the model's prediction was incorrect or poorly calibrated into a key-value pair to be added to the epistemic memory.

**Confidence signal.** The scalar `c_t` derived from the main model's output distribution at each position. The default is the entropy of the next-token distribution.
