from pathlib import Path
import pandas as pd


CLASS_MAP = {
    "real": 0,
    "ai_generated": 1,
    "manipulated": 2,
}


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def collect_images(data_root):
    data_root = Path(data_root)

    records = []

    for class_name, label in CLASS_MAP.items():

        class_dir = data_root / class_name

        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue

        for image_path in class_dir.rglob("*"):

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            records.append({
                "path": str(image_path),
                "class": class_name,
                "label": label,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":

    data_root = "data/raw"

    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_images(data_root)

    if df.empty:
        print("No images found.")
        raise SystemExit(1)

    print("\nDataset summary:")
    print(df["class"].value_counts())

    manifest_path = output_dir / "manifest.csv"

    df.to_csv(manifest_path, index=False)

    print(f"\nManifest saved to: {manifest_path}")
    print(f"Total images: {len(df)}")
