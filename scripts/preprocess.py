from pathlib import Path

import pandas as pd

try:
    from scripts.feature_engineering import (
        TARGET,
        ID_COLUMN,
        engineer_features,
    )
except ModuleNotFoundError:
    from feature_engineering import (
        TARGET,
        ID_COLUMN,
        engineer_features,
    )


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_path = DATA_DIR / "sample_submission.csv"

    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    if TARGET not in train.columns:
        raise ValueError(
            f"Kolom target '{TARGET}' tidak ditemukan."
        )

    if ID_COLUMN not in train.columns:
        raise ValueError(
            f"Kolom '{ID_COLUMN}' tidak ditemukan pada train."
        )

    if ID_COLUMN not in test.columns:
        raise ValueError(
            f"Kolom '{ID_COLUMN}' tidak ditemukan pada test."
        )

    return train, test, sample_submission


def prepare_features(train, test):
    X_train = engineer_features(train)
    X_test = engineer_features(test)

    X_test = X_test.reindex(columns=X_train.columns)

    return X_train, X_test


if __name__ == "__main__":
    train, test, sample = load_data()

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nDistribusi target:")
    print(train[TARGET].value_counts())
    print("\nMissing values train:")
    print(train.isnull().sum().sort_values(ascending=False))
    print("\nContoh sample submission:")
    print(sample.head())
