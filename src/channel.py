"""
Signal generation module for the deep-space interference paper reproduction.
Reference: de Senneville, Ogbe, Towfic (SSC25-RAI-07), Signal Modeling section.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import signal as sp_signal


@dataclass
class SignalConfig:
    # BASE-PAPER FACT: fixed constants, "Constant Values" subsection
    Fs: float = 64_000.0          # sampling frequency (Hz)
    Rs: float = 8_000.0           # symbol rate (bps)
    beta: float = np.pi / 4       # modulation index
    fd_max: float = 200.0         # max frequency shift (Hz)

    @property
    def NSPS(self) -> int:
        # BASE-PAPER FACT: NSPS = Fs / Rs (stated to equal 8 for these constants)
        return int(round(self.Fs / self.Rs))


def generate_bits(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. Bernoulli(p=0.5) bits — BASE-PAPER FACT (Signal Modeling, BPSK)."""
    return rng.integers(0, 2, size=n_symbols)


def nrz_encode(bits: np.ndarray, nsps: int) -> np.ndarray:
    """Eq.(1): bipolar NRZ encoding, upsampled to NSPS samples/symbol."""
    m = 2 * bits - 1                     # 0/1 -> -1/+1
    return np.repeat(m, nsps).astype(float)


def generate_soi(cfg: SignalConfig, n_symbols: int, rng: np.random.Generator):
    """
    Eq.(2)-(3): SOI complex baseband signal with random freq/phase offset.
    Returns (s, bits, fd, phi_m) so callers can use bits for BER scoring later.
    """
    nsps = cfg.NSPS
    n_samples = n_symbols * nsps
    bits = generate_bits(n_symbols, rng)
    m = nrz_encode(bits, nsps)

    n = np.arange(n_samples)
    fd = rng.uniform(-cfg.fd_max, cfg.fd_max)      # OUR ASSUMPTION: paper's fm and fd
                                                     # (Signal Generation Parameters section)
                                                     # refer to the same quantity; using fd_max
                                                     # as the single source of truth throughout
    phi_m = rng.uniform(0, 2 * np.pi)

    s = np.exp(1j * (cfg.beta * m + 2 * np.pi * fd / cfg.Fs * n + phi_m))
    return s, bits, fd, phi_m


def _unit_power_normalize(z: np.ndarray) -> np.ndarray:
    """Interference/noise assumed zero-mean, unit std (Received Signal Model)."""
    power = np.mean(np.abs(z) ** 2)
    return z / np.sqrt(power) if power > 0 else z


def generate_tone(cfg: SignalConfig, n_samples: int, rng: np.random.Generator):
    """Eq.(8)-(9): continuous-wave / single-tone interference."""
    n = np.arange(n_samples)
    fz = rng.uniform(-cfg.Rs, cfg.Rs)
    phi_z = rng.uniform(0, 2 * np.pi)
    z = np.exp(1j * (2 * np.pi * fz / cfg.Fs * n + phi_z))
    return z  # already unit modulus -> unit power, no normalization needed


def generate_square(cfg: SignalConfig, n_samples: int, rng: np.random.Generator):
    """Eq.(10)-(11): square-wave interference."""
    n = np.arange(n_samples)
    fz = rng.uniform(-cfg.Rs, cfg.Rs)
    fs = rng.uniform(0, 2 * cfg.Rs)
    phi_z = rng.uniform(0, 2 * np.pi)
    phi_s = rng.uniform(0, 2 * np.pi)
    z = np.sign(np.sin(2 * np.pi * fs / cfg.Fs * n + phi_s)) * \
        np.exp(1j * (2 * np.pi * fz / cfg.Fs * n + phi_z))
    return z  # unit modulus -> unit power


def generate_chirp(cfg: SignalConfig, n_samples: int, rng: np.random.Generator):
    """Eq.(12)-(13): chirp interference."""
    n = np.arange(n_samples)
    TL = n_samples / cfg.Fs
    fa = rng.uniform(-cfg.Rs, cfg.Rs)
    fb = rng.uniform(-cfg.Rs, cfg.Rs)
    phi_z = rng.uniform(0, 2 * np.pi)
    inst = 2 * np.pi * (fa + (fb - fa) * n / (2 * cfg.Fs * TL)) * n / cfg.Fs + phi_z
    z = np.exp(1j * inst)
    return z  # unit modulus -> unit power


def generate_fawgn(cfg: SignalConfig, n_samples: int, rng: np.random.Generator,
                    filter_order: int = 4, ripple_db: float = 1.0):
    """
    Filtered Additive White Gaussian Noise.
    BASE-PAPER FACT: Chebyshev filter, bandwidth 2*Rs, centered at SOI center freq.
    UNKNOWN (not specified in paper): filter order / ripple -> OUR ASSUMPTION
    (order=4, 1 dB ripple), flagged for later sensitivity check if it matters.
    """
    white = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    nyq = cfg.Fs / 2
    cutoff = cfg.Rs / nyq  # baseband: bandwidth 2*Rs -> lowpass cutoff at Rs
    b, a = sp_signal.cheby1(filter_order, ripple_db, cutoff, btype='low')
    z = sp_signal.lfilter(b, a, white)
    return _unit_power_normalize(z)


INTERFERENCE_GENERATORS = {
    "tone": generate_tone,
    "square": generate_square,
    "chirp": generate_chirp,
    "fawgn": generate_fawgn,
}


def generate_noise(n_samples: int, rng: np.random.Generator):
    """Eq.(7): w ~ CN(0, I) — unit-power complex circular Gaussian."""
    return (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)


def apply_time_shift(s: np.ndarray, cfg: SignalConfig, rng: np.random.Generator):
    """
    Eq.(14)-(15): random time shift within one symbol period.
    OUR ASSUMPTION on mechanics: shift by rolling and truncating edge samples
    (paper defines the shift, not the implementation mechanic).
    """
    k_shift = rng.integers(0, cfg.NSPS + 1)
    return np.roll(s, k_shift), k_shift


def mix_signal(cfg: SignalConfig, n_symbols: int, rng: np.random.Generator,
                interference_type: str | None = "tone",
                contamination: float = 0.5,
                sir_db_range=(-15, 30), snr_db_range=(-15, 30),
                apply_shift: bool = False):
    """
    Eq.(4)-(6): full received-signal mixture x = s + z/sqrt(rho_sir) + w/sqrt(rho_snr).
    Returns dict with x, s (clean), label (interference present), bits, and params
    -- everything downstream detectors/mitigators need.
    """
    s, bits, fd, phi_m = generate_soi(cfg, n_symbols, rng)
    n_samples = len(s)

    if apply_shift:
        s, k_shift = apply_time_shift(s, cfg, rng)
    else:
        k_shift = 0

    has_interference = rng.random() < contamination
    if has_interference and interference_type is not None:
        z = INTERFERENCE_GENERATORS[interference_type](cfg, n_samples, rng)
        sir_db = rng.uniform(*sir_db_range)
        rho_sir = 10 ** (sir_db / 10)
    else:
        z = np.zeros(n_samples, dtype=complex)
        sir_db, rho_sir = None, np.inf

    w = generate_noise(n_samples, rng)
    snr_db = rng.uniform(*snr_db_range)
    rho_snr = 10 ** (snr_db / 10)

    x = s + (z / np.sqrt(rho_sir) if np.isfinite(rho_sir) else 0.0) + w / np.sqrt(rho_snr)

    return {
        "x": x, "s": s, "z": z, "w": w, "bits": bits,
        "label": has_interference, "interference_type": interference_type if has_interference else None,
        "fd": fd, "phi_m": phi_m, "k_shift": k_shift,
        "sir_db": sir_db, "snr_db": snr_db,
    }