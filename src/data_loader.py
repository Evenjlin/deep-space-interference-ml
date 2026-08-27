"""
Dataset assembly: clean (interference-free) samples for training/covariance
estimation, and labeled test sets for AUC evaluation. Moved here from
notebook-local functions in Step 3 for reuse across steps.
"""
import numpy as np
from src.channel import generate_soi, generate_noise, mix_signal


def make_clean_sample(cfg, n_symbols, snr_db, rng):
    """SOI + noise only, no interference -- used to fit detectors/train AE."""
    s, bits, fd, phi_m = generate_soi(cfg, n_symbols, rng)
    w = generate_noise(len(s), rng)
    rho_snr = 10 ** (snr_db / 10)
    return s + w / np.sqrt(rho_snr)


def build_test_set(cfg, n_symbols, rng, interference_type, n_test, sir_db, snr_db):
    """Labeled mixture of interference-present/absent samples for AUC scoring."""
    samples, labels = [], []
    for _ in range(n_test):
        d = mix_signal(cfg, n_symbols, rng, interference_type=interference_type,
                        contamination=0.5,  # BASE-PAPER FACT: test contamination = 50%
                        sir_db_range=(sir_db, sir_db), snr_db_range=(snr_db, snr_db),
                        apply_shift=False)
        samples.append(d["x"]); labels.append(d["label"])
    return samples, np.array(labels, dtype=int)