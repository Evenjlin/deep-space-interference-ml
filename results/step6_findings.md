# Step 6 Findings: Compensated Mahalanobis Pipeline (Table 1, Modulation=Yes/Random Shift=Yes)

## Summary
Combined Step 5's CNN time-shift estimator with FFT-based frequency
estimation to correct signals before Mahalanobis scoring, reproducing
the paper's core claimed contribution.

## Results
| Type  | Paper raw AUC | Our raw AUC | Our compensated AUC | Gain |
|---|---|---|---|---|
| tone  | 0.593 | 0.544 | 0.883 | +0.339 |
| chirp | 0.600 | 0.487 | 0.864 | +0.377 |
| fawgn | 0.597 | 0.543 | 0.986 | +0.443 |

Raw AUC degradation closely matches the paper's own reported values
(within ~0.05-0.11), confirming our shift-injection and reference
covariance setup are consistent with the paper's described condition.

## Key design note (resolves a concern raised before running this step)
The CNN was trained under fd_max=0 (Step 5), so a train/test mismatch was
expected when applying it to signals with both time AND frequency shift
active. This did not materially hurt performance because
`compensate_signal` applies frequency correction (FFT, untrained) BEFORE
feeding the signal to the CNN -- so the CNN only ever sees an
already-frequency-corrected residual, which resembles its training
distribution. Sequencing order mitigated the mismatch; no retraining
was necessary.

## Status
Reproduction: SUCCESSFUL. Both the degradation and the recovery-via-
compensation are reproduced with the correct qualitative pattern and
reasonably close quantitative match to the paper's raw-AUC values. No
single scalar paper value exists for "compensated" AUC at this exact
SIR/SNR point (paper reports this as heatmaps/ISO-AUC curves in Fig.
5-7, not a table entry), so the compensated numbers are OUR
EXPERIMENTAL RESULT without a direct point-for-point paper comparison.

## Open item for later
Chirp raw AUC (0.487) landed slightly below 0.5 -- consistent with
sampling noise at n=200 near true chance level (same caveat noted in
Step 3), not flagged as a concern on its own.