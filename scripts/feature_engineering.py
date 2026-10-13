import pandas as pd

TARGET = "damage_grade"
ID_COLUMN = "building_id"

NUMERIC_COLUMNS = [
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
]

CATEGORICAL_COLUMNS = [
    "district_id",
    "vdcmun_id",
    "ward_id",
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "superstructure",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur yang konsisten untuk train dan test."""

    X = df.drop(
        columns=[ID_COLUMN, TARGET],
        errors="ignore",
    ).copy()

    # Pastikan fitur numerik benar-benar bertipe numerik.
    for col in NUMERIC_COLUMNS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # ID wilayah diperlakukan sebagai kategori, bukan angka kontinu.
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

    # Fitur kategorikal tambahan yang tidak terdaftar tetap ditangani.
    for col in X.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return X
