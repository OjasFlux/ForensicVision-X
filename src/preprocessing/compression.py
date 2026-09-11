from io import BytesIO

from PIL import Image


DEFAULT_QUALITIES = [95, 80, 60, 40, 20, 10]


def jpeg_recompress(image, quality=50):
    """
    Recompress an image as JPEG at the requested quality.
    Returns a new RGB PIL image.
    """
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    buffer = BytesIO()

    image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality)

    buffer.seek(0)

    return Image.open(buffer).convert("RGB").copy()
