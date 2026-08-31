# Step 8a Findings: Denoising-Only Mitigation Autoencoder (Fig. 11/12)

## Summary
Reproduced the paper's own negative-result experiment: an autoencoder
trained only on clean SOI+noise (never shown interference) should fail
to meaningfully reduce BER despite achieving real MSE reduction.

## Bug found and fixed: mode collapse to unconditional mean
Initial implementation collapsed to predicting near-zero output
regardless of input (confirmed via Cell 5c: model output mean-abs=0.018
vs target mean-abs=0.619; loss landed at exactly the MSE of a
zero-output solution, ~0.5, matching mean(target^2), and was unchanged
across a 30x learning-rate sweep, ruling out a slow-convergence
explanation). Root cause: encoder-decoder with no direct input-to-output
path, combined with normalization layers, made "predict the training-set
mean" an easy local optimum. FIXED via a global residual connection
(output = correction + input) added to MitigationAutoencoder.

## Second finding: window length matters for phase-preserving reconstruction
After the collapse fix, BER was still bad on CLEAN signals at N_SYMBOLS=64
(actively WORSE than doing nothing, e.g. mitigated BER exceeded 0.5 at
several Eb/N0 points) -- worse than random guessing. Retraining at
N_SYMBOLS=256 (BASE-PAPER FACT: paper's actual stated window for this
task, which we had not matched) resolved the "worse than chance" symptom:

| Eb/N0 | Mitigated BER (N=64) | Mitigated BER (N=256) |
|---|---|---|
| 0dB | 0.487 | 0.460 |
| 3dB | 0.526 | 0.489 |
| 6dB | 0.527 | 0.489 |

## Remaining gap (OPEN ITEM, not resolved)
Paper's claim is "no effect" -- mitigated BER should track close to raw
interference BER (~0.31-0.33 in our data). Our N=256 result (~0.46-0.49)
is a real improvement over N=64 but still sits between chance (0.50) and
the interference-only line, not matching "no effect" precisely. Not
investigated further at this time -- flagged for possible future work.

## Status
PARTIALLY SUCCESSFUL. Qualitative direction confirmed (denoising-only
training does not restore BER to clean-signal levels), quantitative
match to the paper's specific curve values remains imperfect.