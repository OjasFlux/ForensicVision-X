from torchvision import transforms


def get_training_transform(image_size: int = 224):
    """
    Training augmentation pipeline.
    """

    return transforms.Compose([
        transforms.Resize((256, 256)),

        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
