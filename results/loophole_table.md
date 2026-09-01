# Ranked Loophole Table — Evidence-Backed (Steps 3-8)

| ID | Weakness | Evidence | Severity | Source |
|---|---|---|---|---|
| L1 | Mahalanobis distance has a near-total blind spot for narrowband interference at the SOI's spectral center (AUC craters to ~0.49, chance level) regardless of detection window length (tested 64->256 symbols) | step3_findings.md: frequency-sweep diagnostic, dead zone confirmed at fz=0 across two window lengths | High | Step 3 |
| L2 | MLP autoencoder detector performance is highly sensitive to an architecturally unspecified hyperparameter (bottleneck compression ratio) -- AUC ranged 0.51-0.86 across a 4x sweep | step4_findings.md: bottleneck sweep, monotonic AUC trend | Medium | Step 4 |
| L3 | SNR-corrected covariance (Eq.17) shows an inconsistent, sometimes-negative relationship between estimator accuracy and downstream AUC -- a 3x better-calibrated SNR estimator did not reliably improve, and sometimes hurt, detection AUC vs. plain shift-compensation alone | step7_findings.md: paired v1/v2 grid comparison | High | Step 7 |
| L4 | Mitigation autoencoder architectures (encoder-decoder, no explicit skip path) are prone to collapsing to the unconditional mean when interference/phase structure is unspecified in the paper -- required a global residual connection to avoid | step8a_findings.md | Medium (implementation fragility, not a paper claim per se) | Step 8a |
| L5 | Reconstruction MSE and downstream BER can be directly disconnected: mitigation autoencoder achieved ~7x MSE improvement over raw input at every tested Eb/N0, yet BER was WORSE than no mitigation at all, and got worse (not better) as Eb/N0 increased -- even with interference-aware training (the paper's own proposed fix for this exact phenomenon) | step8b_findings.md: direct MSE-vs-BER diagnostic, ruled out overcorrection | **Very High** | Step 8b |

## Strongest candidate for Research Opportunity (master prompt deliverable #10)

**L5** is the strongest candidate for a genuine follow-on contribution:
- It's the MOST GENERALIZABLE finding -- not specific to this paper's exact
  architecture, but a challenge to a common assumption (MSE as a training
  objective for interference mitigation) used across the wider literature.
- It EXTENDS the paper's own acknowledged limitation (Fig.12's finding for
  denoising-only training) into a regime the paper claims is fixed
  (interference-aware training) -- suggesting the fix may be incomplete.
- It suggests a concrete, testable research direction: a decision-aware
  or perceptually-weighted loss function (e.g., penalizing sign errors
  in the bit-deciding component directly, not just aggregate energy)
  as an alternative training objective.

L1 and L3 are strong secondary candidates, both well-evidenced and
specific to the detection side of the pipeline.