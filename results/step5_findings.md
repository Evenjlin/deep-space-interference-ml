# Step 5 Findings: CNN Time-Shift Estimator

## Summary
Implemented the CNN described in the paper's "Time and Frequency Shift
Estimation" section: 3x Conv1D (kernel=3, stride=2, ReLU+BatchNorm), first
conv with 2 input channels (I/Q) and kernel size=NSPS, global average pool,
FC classifier over NSPS+1=9 possible shift values.

## Ambiguity resolved (OUR ASSUMPTION)
Paper states the network is "trained... using Binary Cross-Entropy Loss"
but also describes a softmax output over NSPS classes -- these are an
unusual pairing (categorical cross-entropy is the standard choice for a
softmax multi-class output). Implemented using `nn.CrossEntropyLoss`
(logits + internal softmax), the standard pairing for this architecture.
Not yet tested against literal per-class BCE as an ablation -- flagged as
an open item, not currently blocking.

## Result (OUR EXPERIMENTAL RESULT)
| Metric | Value |
|---|---|
| Random-guess baseline accuracy (1/9 classes) | 0.1111 |
| Final validation accuracy | 0.8953 |

Training converged fast: val accuracy reached ~0.88 by epoch 3, then
plateaued/oscillated between ~0.883-0.903 through epoch 30 with no further
net improvement. Train/val loss stayed close throughout (no overfitting
gap), both flattening around 0.15-0.16 after epoch ~5.

## Important limitation
BASE-PAPER FACT: the paper does not report a standalone time-shift
classification accuracy number anywhere -- there is nothing to reproduce
this specific figure against. The real validation of "is 0.895 good
enough" is whether it improves downstream compensated-Mahalanobis AUC in
Step 6, not this number in isolation.

## Status
Model trains cleanly, clears the sanity floor (>>2x random chance) by a
wide margin. Considered ready to use as the time-shift correction stage
in the Step 6 compensated-Mahalanobis pipeline. Not independently
validated against a paper-reported benchmark, since none exists for this
component.

## Open items for later
- Ablation: literal per-class BCE loss vs. categorical cross-entropy
  (Experiment 3 from the original loophole audit) -- not yet run.
- Possible mild optimization improvement (lower learning rate / LR
  schedule) to reduce the post-epoch-5 val-accuracy oscillation, if
  Step 6 results suggest this component is a bottleneck.