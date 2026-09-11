from PIL import Image
import pytest

from src.preprocessing.compression import jpeg_recompress


def test_jpeg_recompress():
    image = Image.new("RGB", (640, 480), color="white")

    compressed = jpeg_recompress(image, quality=50)

    assert compressed.mode == "RGB"
    assert compressed.size == (640, 480)


def test_jpeg_quality_range():
    image = Image.new("RGB", (100, 100), color="white")

    with pytest.raises(ValueError):
        jpeg_recompress(image, quality=0)

    with pytest.raises(ValueError):
        jpeg_recompress(image, quality=101)
