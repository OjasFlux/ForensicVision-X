import numpy as np
import cv2


def extract_noise_residual(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Estimate high-frequency noise residual
    using Gaussian smoothing.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image

    gray = np.float32(gray)

    smooth = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        0
    )

    residual = gray - smooth

    return residual


def noise_statistics(
    residual: np.ndarray
) -> dict:

    return {
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "variance": float(np.var(residual)),
        "energy": float(np.mean(residual ** 2))
    }
