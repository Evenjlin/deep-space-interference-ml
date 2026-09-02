# Reproduction, Weakness Analysis, and a Decision-Aware Loss Improvement for ML-Based Interference Detection and Mitigation on Deep Space Telecom Signals

Reproduction and extension of: de Senneville, Ogbe, Towfic — "Machine Learning for
Interference Detection and Mitigation on Deep Space Telecom Signals" (SSC25-RAI-07,
39th Annual Small Satellite Conference).

**Author:** Evenjilin Ekka, B.Tech ECE, NIT Delhi — supervised by Dr. Rikmantra Basu

## Status: Implementation, reproduction, weakness audit, and proposed improvement complete

## Summary

This project independently reproduces the full detection-and-mitigation pipeline of
the base paper, validates it stage by stage against the paper's own reported figures,
identifies five evidence-backed weaknesses through targeted diagnostic experiments,
and proposes and validates a fix for the most significant one — a decision-aware
training loss for the mitigation autoencoder that resolves a disconnect between
reconstruction quality (MSE) and bit-error-rate (BER) outcomes.

**Full write-up:** [`report/Reproduction_and_Decision-Aware_Loss_Improvement_Report.docx`](./report/)

## Project status

| Component | Step(s) | Status |
|---|---|---|
| Signal model (BPSK, 4 interference types, Eq. 1-15) | 2 | Done |
| Mahalanobis + PCA detectors (Fig. 3) | 3 | Done — `results/step3_findings.md` |
| MLP Autoencoder detector (Fig. 3) | 4 | Done — `results/step4_findings.md` |
| CNN time-shift estimator | 5 | Done — `results/step5_findings.md` |
| FFT freq estimation + compensated Mahalanobis (Table 1) | 6 | Done — `results/step6_findings.md` |
| SNR estimation (Eq. 17) + full ISO-AUC grid (Fig. 5-7) | 7 | Done — `results/step7_findings.md` |
| Mitigation autoencoder: denoising-only (Fig. 11/12) | 8a | Done — `results/step8a_findings.md` |
| Mitigation autoencoder: interference-aware, tone/square/chirp (Fig. 13-15) | 8b/8c | Done — `results/step8b_findings.md`, `step8c_findings.md` |
| **Proposed improvement: decision-aware loss** | 9 | Done — `results/step9_findings.md` |
| Final write-up | — | Done — `report/` |

## Key findings

1. **Mahalanobis distance has a frequency-dependent blind spot** — near-total detection
   failure (AUC≈0.49) for tone interference near the SOI's spectral center, not
   surfaced by the paper's pooled AUC metric. (`step3_findings.md`)
2. **MLP autoencoder detection is highly sensitive to an unspecified bottleneck
   hyperparameter** — AUC ranged 0.51-0.86 across a 4x sweep. (`step4_findings.md`)
3. **SNR-corrected covariance (Eq. 17) shows accuracy/benefit disconnect** — a 3x
   better-calibrated SNR estimator did not reliably improve, and sometimes hurt,
   downstream detection AUC. (`step7_findings.md`)
4. **Mitigation autoencoder architectures collapse to the unconditional mean**
   without an explicit residual connection — found and fixed. (`step8a_findings.md`)
5. **MSE reconstruction quality is disconnected from BER outcomes** (primary
   finding) — mitigation autoencoders achieved ~7x MSE improvement over raw input
   at every tested Eb/N0, yet produced BER *worse* than no mitigation at all,
   across all three tested interference types (tone/square/chirp), even with the
   paper's own proposed fix (interference-aware training). (`step8b/8c_findings.md`)
6. **Proposed fix validated**: a decision-aware loss (MSE + hinge-margin penalty on
   the decision-critical de-rotated signal component) outperformed both no-mitigation
   and MSE-only baselines in all 21 tested Eb/N0 points, across all 3 interference
   types. (`step9_findings.md`)

Full ranked loophole table: [`results/loophole_table.md`](./results/loophole_table.md)

## Repo structure

- `src/` — reusable modules: `channel.py` (signal generation), `model.py` (detectors,
  CNNs, autoencoders), `data_loader.py`, `train.py`, `evaluate.py` (BER scoring)
- `notebooks/` — one notebook per step, self-contained (each clones/installs
  independently), run in Google Colab
- `results/` — per-step findings docs (`.md`) and metrics (`.csv`)
- `figures/` — generated plots
- `checkpoints/` — trained model weights
- `report/` — final write-up (Word document)

## Reproducing this work

Every notebook is self-contained: open in Colab, run the bootstrap cell (device
check, repo clone/pull, dependency install), then run cells in order. See each
notebook's early cells for the exact hyperparameters and BASE-PAPER FACT /
OUR ASSUMPTION labeling used throughout.