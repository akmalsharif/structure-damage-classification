from pathlib import Path

import joblib
import pandas as pd

try:
    from scripts.preprocess import load_data, prepare_features
    from scripts.feature_engineering import ID_COLUMN
except ModuleNotFoundError:
    from preprocess import load_data, prepare_features
    from feature_engineering import ID_COLUMN


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "pipeline_model_lightgbm.pkl"
OUTPUT_DIR = ROOT_DIR / "submissions"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model belum tersedia di {MODEL_PATH}. Jalankan script training terlebih dahulu."
        )

    print("1. Memuat model dan data...")
    artifact = joblib.load(MODEL_PATH)

    train, test, sample = load_data()
    _, X_test = prepare_features(train, test)

    # Samakan fitur dengan fitur saat training.
    feature_columns = artifact["feature_columns"]
    X_test = X_test.reindex(columns=feature_columns)

    for col in artifact.get("categorical_features", []):
        if col in X_test.columns:
            categories = artifact["categorical_categories"][col]
            X_test[col] = pd.Categorical(
                X_test[col].astype("string"),
                categories=categories,
            )

    model = artifact["model"]
    print("2. Membuat prediksi...")

    predictions = model.predict(X_test)

    predictions = (
        pd.Series(predictions)
        .astype(str)
        .str.replace("Grade ", "", regex=False)
        .values
    )

    submission = pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN].values,
        "damage_grade": predictions,
    })

    # Validasi format dan jumlah baris.
    assert len(submission) == len(test)
    assert submission[ID_COLUMN].equals(test[ID_COLUMN])
    assert submission.columns.tolist() == [
        "building_id",
        "damage_grade",
    ]
    assert submission["damage_grade"].notna().all()

    # Validasi kelas target terhadap sample_submission
    expected_classes = set(
        sample["damage_grade"].astype(str).str.replace("Grade ", "", regex=False)
    )
    predicted_classes = set(
        submission["damage_grade"].astype(str).str.replace("Grade ", "", regex=False)
    )

    if expected_classes and not predicted_classes.issubset(expected_classes):
        raise ValueError(
            "Label prediksi tidak sesuai dengan label pada sample_submission.csv."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print("\nSubmission berhasil dibuat.")
    print("Lokasi:", OUTPUT_PATH)
    print("\nPreview:")
    print(submission.head())
    print("\nDistribusi prediksi:")
    print(submission["damage_grade"].value_counts())
    print("\nUkuran submission:", submission.shape)


if __name__ == "__main__":
    main()