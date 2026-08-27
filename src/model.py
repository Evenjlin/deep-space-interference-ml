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