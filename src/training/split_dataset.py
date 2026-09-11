from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


SEED = 42


def split_dataset(
    manifest_path,
    output_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
):

    assert abs(
        train_ratio + val_ratio + test_ratio - 1.0
    ) < 1e-6

    df = pd.read_csv(manifest_path)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=SEED,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df.to_csv(
        output_dir / "val.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    print("Dataset split completed.")

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["class"].value_counts())

    print("\nValidation distribution:")
    print(val_df["class"].value_counts())

    print("\nTest distribution:")
    print(test_df["class"].value_counts())


if __name__ == "__main__":

    split_dataset(
        manifest_path="data/splits/manifest.csv",
        output_dir="data/splits",
    )
