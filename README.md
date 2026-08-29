# Deep Space Interference Detection & Mitigation — Reproduction & Loophole Audit

Reproduction of: de Senneville, Ogbe, Towfic — "Machine Learning for Interference
Detection and Mitigation on Deep Space Telecom Signals" (SSC25-RAI-07).

## Status

| Component | Step | Status |
|---|---|---|
| Signal model (BPSK, 4 interference types, Eq. 1-15) | 2 | Done |
| Mahalanobis + PCA detectors (Fig. 3) | 3 | Done — see results/step3_findings.md |
| MLP Autoencoder detector (Fig. 3) | 4 | Done — see results/step4_findings.md |
| CNN time-shift estimator | 5 | Done — see results/step5_findings.md |
| FFT freq estimation + compensated Mahalanobis (Table 1) | 6 | Done — see results/step6_findings.md |
| SNR estimation (Eq. 17) + full ISO-AUC grids (Fig. 5-7) | 7 | In progress |
| CNN Autoencoder mitigation (Fig. 11-15) | 8+ | Not started |

## Key findings so far
- Mahalanobis distance has a strong frequency-dependent blind spot for
  tone interference near the SOI's spectral center — not surfaced by the
  paper's single pooled AUC metric (see step3_findings.md).
- MLP autoencoder detector is highly sensitive to bottleneck compression
  ratio, an unspecified hyperparameter (see step4_findings.md).
- Shift compensation (CNN time-shift + FFT frequency estimation) recovers
  most of the AUC lost to time/frequency shifts, confirming the paper's
  core claimed mechanism (see step6_findings.md).

## Repo structure
- `src/` — reusable modules (channel.py, model.py, data_loader.py, train.py)
- `notebooks/` — one notebook per step, run in Google Colab
- `results/` — findings docs (.md) and metrics (.csv) per step
- `figures/` — generated plots
- `checkpoints/` — trained model weights