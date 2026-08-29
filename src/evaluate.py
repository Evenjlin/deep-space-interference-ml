"""BER evaluation for the mitigation autoencoder, per the paper's matched-filter methodology."""
import numpy as np
from scipy.special import erfc


def theoretical_ber(eb_n0_db: float) -> float:
    """Eq.(20): BER = 0.5*erfc(sqrt(Eb/N0))"""
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    return 0.5 * erfc(np.sqrt(eb_n0_linear))


def snr_db_to_ebn0_db(snr_db: float, nsps: int) -> float:
    """OUR ASSUMPTION: standard derivation, not given explicitly in paper."""
    return snr_db + 10 * np.log10(nsps)


def matched_filter_ber(x_hat: np.ndarray, bits_true: np.ndarray, cfg, n_symbols: int,
                         fd: float, phi_m: float, k_shift: int) -> float:
    """
    OUR ASSUMPTION on exact arithmetic: de-rotate by TRUE fd/phi_m/k_shift
    (BASE-PAPER FACT: paper assumes perfect downstream tracking), block-average
    each symbol, decide via sign of IMAGINARY part (only component that
    discriminates beta*m = +-pi/4). Discards first/last symbol per BASE-PAPER FACT.
    """
    nsps = cfg.NSPS
    n = np.arange(len(x_hat))
    x_derot = np.roll(x_hat, -k_shift) * np.exp(-1j * (2*np.pi*fd/cfg.Fs*n + phi_m))
    symbols = x_derot[:n_symbols*nsps].reshape(n_symbols, nsps).mean(axis=1)
    bits_hat = (symbols.imag > 0).astype(int)
    bits_hat_trimmed, bits_true_trimmed = bits_hat[1:-1], bits_true[1:-1]
    return np.sum(bits_hat_trimmed != bits_true_trimmed) / len(bits_true_trimmed)