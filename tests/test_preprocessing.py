from PIL import Image
import torch

from src.preprocessing.image_loader import (
    load_image,
    get_inference_transform,
)


def test_load_image(tmp_path):
    image_path = tmp_path / "test.jpg"

    image = Image.new("RGB", (640, 480), color="white")
    image.save(image_path)

    loaded = load_image(image_path)

    assert loaded.mode == "RGB"
    assert loaded.size == (640, 480)


def test_inference_transform():
    image = Image.new("RGB", (640, 480), color="white")

    transform = get_inference_transform()
    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
