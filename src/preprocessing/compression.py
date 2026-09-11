from io import BytesIO
from PIL import Image


def jpeg_compress(
    image: Image.Image,
    quality: int
) -> Image.Image:
    """
    Compress an image using JPEG and return
    the decompressed image.
    """

    if not 1 <= quality <= 100:
        raise ValueError(
            "JPEG quality must be between 1 and 100."
        )

    image = image.convert("RGB")

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")

    return compressed
  def generate_compression_levels(image):
    """
    Generate multiple JPEG compression versions.
    """

    qualities = [95, 80, 60, 40, 20, 10]

    return {
        quality: jpeg_compress(image, quality)
        for quality in qualities
    }
