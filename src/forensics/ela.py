from io import BytesIO
import numpy as np
from PIL import Image, ImageChops


def compute_ela(
    image: Image.Image,
    quality: int = 90
) -> Image.Image:
    """
    Compute Error Level Analysis (ELA).
    """

    image = image.convert("RGB")

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    recompressed = Image.open(buffer).convert("RGB")

    difference = ImageChops.difference(
        image,
        recompressed
    )

    extrema = difference.getextrema()

    max_difference = max(
        channel_max
        for _, channel_max in extrema
    )

    if max_difference == 0:
        max_difference = 1

    scale = 255.0 / max_difference

    ela = difference.point(
        lambda pixel: int(pixel * scale)
    )

    return ela
