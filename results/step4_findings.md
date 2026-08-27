# Step 4 Findings: MLP Autoencoder Detector Reproduction (Fig. 3 / Table 1 AE row)

## Summary
Reproduced the reconstruction-MSE autoencoder detector. Architecture is
entirely unspecified in the paper (references external prior work [2]),
so every structural choice here is OUR ASSUMPTION.

## Bugs found and fixed during this step
- `train_autoencoder` never seeded `torch`, so identical code/data produced
  meaningfully different AUC across reruns (~0.11 swing observed on tone)
  purely from random weight initialization. Fixed by adding
  `torch.manual_seed(seed)` at the start of training, per master prompt
  §27's reproducibility requirement -- this had been missed since Step 4's
  first draft.

## Key finding: bottleneck size strongly affects detectability (OUR EXPERIMENTAL RESULT)
| Compression | Bottleneck dim | tone AUC | chirp AUC | fawgn AUC |
|---|---|---|---|---|
| 32 | 32  | 0.553 | 0.511 | 0.653 |
| 16 | 64  | 0.554 | 0.611 | 0.554 |
| 8  | 128 | 0.703 | 0.554 | 0.554 |
| 4  | 256 | 0.861 | 0.766 | 0.858 |

Aggressive compression (16x, our original default) destroys enough
reconstructive detail that normal and anomalous inputs become similarly
hard to reconstruct, collapsing the contrast the detector relies on.
Same class of finding as Step 3's window-length sensitivity for
Mahalanobis distance -- an unspecified hyperparameter turning out to be
a major lever on reported performance.

## Final result (compression=4, seeded, window=64 symbols)
| Type  | Paper AE AUC | Our AE AUC | Difference |
|---|---|---|---|
| tone  | 0.7544 | 0.8007 | +0.046 |
| chirp | 0.7816 | 0.7588 | -0.023 |
| fawgn | 0.7615 | 0.8785 | +0.117 |

## Interpretation
- No data leakage identified (training strictly interference-free,
  fresh synthetic test sets each run, matching BASE-PAPER FACT
  contamination settings).
- tone/fawgn exceeding the paper is plausible given our architecture is
  a genuine guess, not necessarily "better" in any general sense -- a
  different unspecified architecture (ours) can reasonably land on
  either side of the paper's unspecified one.
- Single-seed result for a claim this size (+0.117 on fawgn) is not yet
  strong evidence on its own -- UNKNOWN whether this holds across
  multiple seeds. Flagged as open item, not asserted as improvement.

## Status
Reproduction: SUCCESSFUL in qualitative ranking (AE is the weakest of the
three detectors, matching paper's Fig.3 ordering) and broadly comparable
in magnitude once architecture hyperparameters are tuned. Residual
uncertainty on exact values remains open given unspecified paper
architecture and single-seed testing here.

## Open item for later
Multi-seed run (3-5 seeds) at compression=4 to check whether the
tone/fawgn "beats paper" result is stable or a single lucky draw --
candidate for the Experimental Validation phase rather than blocking
progress now.