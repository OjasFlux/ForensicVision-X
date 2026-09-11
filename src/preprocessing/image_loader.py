from pathlib import Path
from PIL import Image


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(image_path: str) -> Image.Image:
    """
    Load an image and convert it to RGB.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}"
        )

    image = Image.open(path).convert("RGB")

    return image
  from torchvision import transforms


def get_inference_transform(image_size: int = 224):
    """
    Transformation used during model inference.
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
