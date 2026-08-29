"""
Closed-form/statistical anomaly detectors: Mahalanobis distance (Eq. 16) and
PCA reconstruction error. No training/gradient descent -- both are fit by
direct computation on a pool of interference-free samples.
"""
import numpy as np
from sklearn.decomposition import PCA


def vectorize(x: np.ndarray) -> np.ndarray:
    """
    Complex I/Q vector -> real 2N-dim vector (real || imag).
    OUR ASSUMPTION: paper's kappa(x) = x^T C^-1 x uses transpose, not
    conjugate-transpose -- implies real vector representation.
    """
    return np.concatenate([x.real, x.imag])


def estimate_covariance(clean_samples: list, ridge: float = 1e-6) -> np.ndarray:
    """
    C = E{x x^T} over pure SOI+noise samples (no interference).
    BASE-PAPER FACT: 'C denotes the covariance matrix associated with the
    SOI and noise alone' (accounting for imperfections like shifts).
    ridge: OUR ASSUMPTION, small diagonal loading for numerical invertibility.
    """
    vecs = np.stack([vectorize(x) for x in clean_samples])  # (M, 2N)
    C = (vecs.T @ vecs) / vecs.shape[0]
    C += ridge * np.eye(C.shape[0])
    return C


def mahalanobis_score(x: np.ndarray, C_inv: np.ndarray) -> float:
    """Eq. (16): kappa(x) = x^T C^-1 x"""
    v = vectorize(x)
    return float(v @ C_inv @ v)


def fit_pca_detector(clean_samples: list, variance_threshold: float = 0.95) -> PCA:
    """
    Fit PCA on interference-free samples. n_components chosen to explain
    `variance_threshold` of variance -- NOT SPECIFIED in the paper (OUR ASSUMPTION).
    """
    vecs = np.stack([vectorize(x) for x in clean_samples])
    pca_full = PCA().fit(vecs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    return PCA(n_components=n_components).fit(vecs)


def pca_score(x: np.ndarray, pca: PCA) -> float:
    """Reconstruction error after projecting onto/back from the fitted subspace."""
    v = vectorize(x).reshape(1, -1)
    v_proj = pca.inverse_transform(pca.transform(v))
    return float(np.sum((v - v_proj) ** 2))


import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """
    OUR ASSUMPTION: architecture not specified in paper (external ref [2]).
    Simple 3-layer encoder/decoder; bottleneck size configurable via `compression`.
    """
    def __init__(self, input_dim: int, compression: int = 16):
        super().__init__()
        bottleneck = max(8, input_dim // compression)
        hidden = input_dim // 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def ae_score(x: np.ndarray, model: MLPAutoencoder, device) -> float:
    """Reconstruction MSE as anomaly score -- BASE-PAPER FACT for AE detector."""
    v = torch.tensor(vectorize(x), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        recon = model(v)
        mse = torch.mean((v - recon) ** 2).item()
    return mse


class TimeShiftCNN(nn.Module):
    """
    BASE-PAPER FACT (architecture): 3x Conv1D (kernel=3, stride=2, ReLU, BatchNorm),
    first conv has 2 input channels (I/Q) and kernel size = NSPS, global average
    pool over time, FC + softmax over NSPS classes.
    OUR ASSUMPTION: paper says "Binary Cross-Entropy" but describes a softmax
    multi-class output -- implemented here as raw logits for nn.CrossEntropyLoss
    (mathematically the standard pairing for this architecture).
    """
    def __init__(self, nsps: int):
        super().__init__()
        self.nsps = nsps
        self.conv1 = nn.Conv1d(2, 16, kernel_size=nsps, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, nsps + 1)  # classes: 0..NSPS inclusive (Eq.15)

    def forward(self, x):
        # x: (batch, 2, time) -- real/imag as channels
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)  # global average pool over time
        return self.fc(x)  # raw logits -- CrossEntropyLoss applies softmax internally


def complex_to_channels(x: np.ndarray) -> np.ndarray:
    """(N,) complex -> (2, N) real/imag stacked as channels, for CNN input."""
    return np.stack([x.real, x.imag])


def estimate_frequency_shift(x: np.ndarray, cfg) -> float:
    """
    BASE-PAPER FACT: 'argmax of FFT within main lobe' estimates fd.
    Main lobe = [-Rs, Rs] per the paper's stated search range.
    """
    N = len(x)
    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1/cfg.Fs)
    mask = (freqs >= -cfg.Rs) & (freqs <= cfg.Rs)
    peak_idx = np.argmax(np.abs(spectrum[mask]))
    return freqs[mask][peak_idx]


def compensate_signal(x: np.ndarray, cfg, timeshift_model, device) -> np.ndarray:
    """
    Estimate time shift (CNN) and frequency shift (FFT argmax), then
    correct the signal by undoing both -- OUR ASSUMPTION on mechanics,
    since the paper describes what is estimated but not the exact
    correction procedure.
    """
    import torch
    # Frequency correction
    fd_hat = estimate_frequency_shift(x, cfg)
    n = np.arange(len(x))
    x_freq_corrected = x * np.exp(-1j * 2 * np.pi * fd_hat / cfg.Fs * n)

    # Time-shift correction (CNN inference)
    channels = complex_to_channels(x_freq_corrected).astype(np.float32)
    with torch.no_grad():
        inp = torch.tensor(channels, device=device).unsqueeze(0)
        logits = timeshift_model(inp)
        k_hat = int(logits.argmax(dim=1).item())
    x_corrected = np.roll(x_freq_corrected, -k_hat)

    return x_corrected, fd_hat, k_hat


class SNRRegressionCNN(nn.Module):
    """
    OUR ASSUMPTION: paper says 'same CNN employed for time shift estimation'
    but that network has a 9-way softmax output, structurally incompatible
    with regression. Implemented as a separate network sharing the same
    conv backbone shape, with a single-value regression head instead.
    """
    def __init__(self, nsps: int):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=nsps, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, 1)  # regression: predicts SNR in dB

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        return self.fc(x).squeeze(-1)


def corrected_covariance(C_soi: np.ndarray, C_noise: np.ndarray, snr_db_hat: float) -> np.ndarray:
    """Eq.(17): C(SNR) = (rho_snr*C_SOI + C_noise) / (rho_snr + 1)"""
    rho_snr = 10 ** (snr_db_hat / 10)
    return (rho_snr * C_soi + C_noise) / (rho_snr + 1)