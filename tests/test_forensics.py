import numpy as np

from src.forensics.fft import compute_fft
from src.forensics.dct import (
    compute_dct,
    dct_statistics
)
from src.forensics.noise import (
    extract_noise_residual,
    noise_statistics
)


def test_fft():

    image = np.random.randint(
        0,
        256,
        (224, 224, 3),
        dtype=np.uint8
    )

    result = compute_fft(image)

    assert result.shape == (
        224,
        224
    )


def test_dct():

    image = np.random.randint(
        0,
        256,
        (224, 224, 3),
        dtype=np.uint8
    )

    dct = compute_dct(image)

    stats = dct_statistics(dct)

    assert "mean" in stats
    assert "std" in stats


def test_noise():

    image = np.random.randint(
        0,
        256,
        (224, 224, 3),
        dtype=np.uint8
    )

    residual = extract_noise_residual(
        image
    )

    stats = noise_statistics(
        residual
    )

    assert "variance" in stats
