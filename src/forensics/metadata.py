from PIL import Image


def extract_metadata(image_path: str) -> dict:
    """
    Extract basic image metadata.
    """

    image = Image.open(image_path)

    metadata = {
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height
    }

    exif = image.getexif()

    metadata["exif_available"] = len(exif) > 0

    return metadata
