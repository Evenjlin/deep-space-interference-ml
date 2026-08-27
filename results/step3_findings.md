# Step 3 Findings: Mahalanobis Detector Reproduction (Fig. 3, Tone Interference)

## Summary
Reproduced Mahalanobis distance detection (Eq. 16) at SIR=20dB, SNR=15dB, no
time/frequency shift (Fig. 3 condition). Found a frequency-dependent blind
spot not visible in the paper's pooled AUC reporting.

## Results

| Detection window | Pooled tone AUC (ours) | Paper's reported tone AUC |
|---|---|---|
| 16 symbols  | 0.8813 | 0.9861 |
| 64 symbols  | 0.8741 | 0.9861 |
| 256 symbols | 0.9502 | 0.9861 |

## Key finding (OUR EXPERIMENTAL RESULT)
AUC vs. tone frequency (fz) at window=256 symbols:

| fz (Hz) | AUC |
|---|---|
| -8000 to -4000 | ~1.00 |
| -2000 | 0.977 |
| 0 | **0.491** |
| 2000 | 0.976 |
| 4000 to 8000 | ~1.00 |

A tone at fz=0 (the SOI's own spectral center) is essentially undetectable
by Mahalanobis distance regardless of detection window length (0.4862 at
window=64, 0.4905 at window=256 -- no improvement with 4x more data).
This is structural: fz=0 is the direction of highest natural SOI variance,
so added interference energy there doesn't register as anomalous relative
to the covariance matrix C.

## Interpretation
- The paper's Eq. (9) draws tone frequency uniformly over the full
  main lobe, U(-Rs, Rs) -- the same distribution we tested against.
- A wider detection window substantially narrows the affected frequency
  range (dead zone shrinks from ~±4000Hz at window=64 to a narrow spike
  at window=256), closing most (~60%) of the gap to the paper's reported
  AUC.
- A residual gap (~0.036 at window=256) remains unexplained -- candidate
  causes: ridge regularization value (OUR ASSUMPTION), possible narrower
  fz sampling range in the paper's actual test than Eq.(9) implies
  (UNKNOWN), or a longer window than 256 symbols (UNTESTED).

## Status
Reproduction: PARTIALLY SUCCESSFUL. Qualitative ranking (Mahalanobis > PCA)
matches the paper. Quantitative AUC has a real, diminishing-but-present
gap, primarily explained by detection window length, with the fz=0 blind
spot identified as a genuine limitation of the Mahalanobis approach not
surfaced by the paper's single pooled AUC metric.

## Candidate loophole for later experimentation
**L-freq-blindspot**: Mahalanobis distance detectability is strongly
frequency-dependent for narrowband interference near the SOI's spectral
center; a single pooled AUC number obscures this and may not reflect
worst-case performance. Severity: Medium-High. Testable further via:
targeted evaluation restricted to fz within +/-1000Hz of center, across
all four interference types, to check whether chirp/square/fawgn share
this vulnerability or whether it's tone-specific (their broadband/swept
nature may make them naturally immune).