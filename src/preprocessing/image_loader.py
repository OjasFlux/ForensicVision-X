from pathlib import Path

from PIL import Image
from torchvision import transforms


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path):
    """
    Load an image and convert it to RGB.
    """
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    return Image.open(path).convert("RGB")


def get_inference_transform(image_size=224):
    """
    Preprocessing pipeline for model inference.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
