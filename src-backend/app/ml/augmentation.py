import numpy as np

from app.models.training import AugmentationConfig


def augment(
    X: np.ndarray, y: np.ndarray, config: AugmentationConfig
) -> tuple[np.ndarray, np.ndarray]:
    if not config.enabled:
        return X, y

    parts_X = [X]
    parts_y = [y]

    if config.gaussian_noise_sigma > 0:
        noise = np.random.normal(0, config.gaussian_noise_sigma, X.shape).astype(X.dtype)
        parts_X.append(X + noise)
        parts_y.append(y)

    if config.bootstrap_enabled:
        indices = np.random.choice(len(X), size=len(X), replace=True)
        parts_X.append(X[indices])
        parts_y.append(y[indices])

    return np.concatenate(parts_X), np.concatenate(parts_y)
