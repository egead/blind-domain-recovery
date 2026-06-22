import numpy as np


def recover_phase(y, channel=0):
    y = np.asarray(y)
    if y.ndim == 3:
        profiles = y[:, :, channel]
    elif y.ndim == 2:
        profiles = y
    else:
        raise ValueError("Expected model output of shape (batch, positions[, channels]).")

    profiles = profiles - np.mean(profiles, axis=1, keepdims=True)
    fft = np.fft.rfft(profiles, axis=1)
    fundamental = fft[:, 1]
    return np.mod(-np.angle(fundamental), 2.0 * np.pi)


def circular_correlation(alpha, beta):
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    alpha_bar = _circular_mean(alpha)
    beta_bar = _circular_mean(beta)

    da = np.sin(alpha - alpha_bar)
    db = np.sin(beta - beta_bar)

    num = np.sum(da * db)
    den = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2)) + 1e-12
    return num / den


def best_circular_correlation(alpha, beta):
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    candidates = {
        "direct": circular_correlation(alpha, beta),
        "reflected": circular_correlation(alpha, -beta),
    }
    best_key = max(candidates, key=lambda k: abs(candidates[k]))
    return abs(candidates[best_key]), best_key, candidates


def _circular_mean(angles):
    return np.angle(np.mean(np.exp(1j * angles)))
