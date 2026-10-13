from pathlib import Path

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from scripts.feature_engineering import TARGET
from scripts.preprocess import load_data, prepare_features


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "pipeline_model_lightgbm.pkl"
MAX_TUNING_ROWS = 100_000
N_TRIALS = 30
N_CV_SPLITS = 3
RANDOM_STATE = 42


def main():
    print("1. Memuat data...")
    train, test, _ = load_data()

    X, _ = prepare_features(train, test)
    y = train[TARGET].astype(str)

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    X[categorical_features] = X[categorical_features].astype("category")

    print("\nDistribusi target:")
    print(y.value_counts())
    print("\nJumlah baris:", len(X))
    print("Jumlah fitur:", X.shape[1])
    print("Fitur kategorikal:", categorical_features)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if len(X_train) > MAX_TUNING_ROWS:
        X_tuning, _, y_tuning, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TUNING_ROWS,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
    else:
        X_tuning, y_tuning = X_train, y_train
    print(f"\nData untuk tuning: {len(X_tuning)} baris")

    cv = StratifiedKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1800, step=100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.015, 0.12, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 120, step=10
            ),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
        model = LGBMClassifier(
            objective="multiclass",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
            subsample_freq=1,
            **params,
        )
        scores = cross_val_score(
            model,
            X_tuning,
            y_tuning,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        return float(scores.mean())

    print("\n2. Tuning LightGBM dengan Optuna TPE untuk Macro F1...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\nParameter terbaik:")
    print(study.best_params)
    print(f"Best {N_CV_SPLITS}-fold CV Macro F1: {study.best_value:.6f}")

    print("\n3. Melatih model terbaik pada seluruh data training...")
    model = LGBMClassifier(
        objective="multiclass",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
        subsample_freq=1,
        **study.best_params,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_features)

    print("\n4. Evaluasi validasi...")
    y_pred = model.predict(X_valid)
    class_labels = model.classes_
    macro_f1 = f1_score(
        y_valid,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(f"\nValidation Macro F1-Score: {macro_f1:.6f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_valid,
            y_pred,
            labels=class_labels,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_valid, y_pred, labels=class_labels)
    pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels,
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "categorical_features": categorical_features,
        "categorical_categories": {
            column: X_train[column].cat.categories.tolist()
            for column in categorical_features
        },
        "best_params": study.best_params,
        "best_cv_macro_f1": float(study.best_value),
        "validation_macro_f1": float(macro_f1),
        "tuning_method": "Optuna TPE",
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel tersimpan di: {MODEL_PATH}")
    print("\nTraining selesai.")


if __name__ == "__main__":
    main()
