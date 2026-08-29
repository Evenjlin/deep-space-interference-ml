# Checkpoint Inventory

| File | Purpose | Status |
|---|---|---|
| ae_detector_n64.pt | Original MLP autoencoder, 16x bottleneck, unseeded | Superseded — kept for history |
| ae_detector_n64_compression4_seeded.pt | Final AE: 4x bottleneck, seeded | **Canonical** (Step 4 result) |
| timeshift_cnn.pt | CNN time-shift classifier | **Canonical** (Step 5) |
| snr_cnn.pt | SNR regressor v1, trained on clean signals only | Kept for comparison — see step7_findings.md |
| snr_cnn_v2_interference_aware.pt | SNR regressor v2, trained with interference present | Better-calibrated (Step 7b), but does NOT reliably improve downstream AUC over v1 — see step7_findings.md. Neither is unambiguously "canonical"; both are referenced in the findings. |