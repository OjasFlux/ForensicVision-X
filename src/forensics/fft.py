import numpy as np
import cv2


def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered FFT magnitude spectrum.

    Parameters
    ----------
    image : np.ndarray
        RGB or grayscale image.

    Returns
    -------
    np.ndarray
        Log-scaled FFT magnitude spectrum.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image

    gray = np.float32(gray)

    fft = np.fft.fft2(gray)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    spectrum = np.log1p(magnitude)

    spectrum = cv2.normalize(
        spectrum,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return spectrum.astype(np.uint8)
