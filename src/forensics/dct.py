import numpy as np
import cv2


def compute_dct(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D DCT of a grayscale image.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image

    gray = np.float32(gray)

    gray -= 128.0

    dct = cv2.dct(gray)

    return dct


def dct_statistics(dct: np.ndarray) -> dict:
    """
    Extract basic statistical information
    from DCT coefficients.
    """

    abs_dct = np.abs(dct)

    return {
        "mean": float(np.mean(abs_dct)),
        "std": float(np.std(abs_dct)),
        "max": float(np.max(abs_dct)),
        "energy": float(np.mean(dct ** 2))
    }
