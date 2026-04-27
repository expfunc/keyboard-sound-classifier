"""Train baseline models for keyboard sound classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import (
    BEST_CNN_MODEL_PATH,
    BEST_MODEL_PATH,
    EVALUATION_BUNDLE_PATH,
    LABEL_ENCODER_PATH,
    MODEL_CANDIDATES_DIR,
    MODEL_INFO_PATH,
    MODEL_METRICS_PATH,
    MODEL_REGISTRY_PATH,
    PROCESSED_DATA_DIR,
)
from src.features import extract_features, extract_mel_spectrogram


def collect_dataset_entries(processed_dir: str | Path = PROCESSED_DATA_DIR) -> list[tuple[Path, str]]:
    """Collect processed WAV file paths with their labels."""
    processed_root = Path(processed_dir)
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_root}")

    wav_files = sorted(processed_root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No processed WAV files found in: {processed_root}")

    entries: list[tuple[Path, str]] = []
    for file_path in wav_files:
        label = file_path.parent.name
        entries.append((file_path, label))

    return entries


def build_tabular_dataset(entries: list[tuple[Path, str]]) -> pd.DataFrame:
    """Extract classic tabular audio features for sklearn models."""
    rows: list[dict[str, object]] = []
    for file_path, label in entries:
        feature_vector = extract_features(file_path)
        row = {f"feature_{index}": value for index, value in enumerate(feature_vector)}
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def build_mel_dataset(entries: list[tuple[Path, str]]) -> np.ndarray:
    """Extract mel-spectrogram tensors for CNN training."""
    mel_tensors = [extract_mel_spectrogram(file_path) for file_path, _ in entries]
    return np.stack(mel_tensors, axis=0)


def build_models() -> dict[str, Pipeline]:
    """Create candidate baseline models."""
    return {
        "RandomForest": Pipeline(
            steps=[
                ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
            ]
        ),
        "SVC": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "KNeighbors": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def build_cnn_model(input_shape: tuple[int, ...], num_classes: int) -> object:
    """Create a compact CNN for mel-spectrogram classification."""
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("TensorFlow is required for the mel-spectrogram CNN. Install `tensorflow-cpu`.") from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def candidate_model_path(model_name: str, model_type: str) -> Path:
    """Return a stable save path for a candidate model."""
    MODEL_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    slug = (
        model_name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .lower()
    )
    suffix = ".keras" if model_type == "cnn" else ".joblib"
    return MODEL_CANDIDATES_DIR / f"{slug}{suffix}"


def train_and_select_best(processed_dir: str | Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Train candidate models, save the best one, and return a metrics table."""
    entries = collect_dataset_entries(processed_dir)
    labels = [label for _, label in entries]
    if len(set(labels)) < 2:
        raise ValueError("Training requires at least two different labels.")

    dataset = build_tabular_dataset(entries)
    X = dataset.drop(columns=["label"])
    y_text = np.asarray(labels)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    sample_indices = np.arange(len(entries))

    try:
        train_indices, test_indices, y_train, y_test = train_test_split(
            sample_indices,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified train/test split. "
            "Record more samples for each label, ideally at least 5 per class."
        ) from exc

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    train_label_counts = (
        pd.Series(label_encoder.inverse_transform(y_train), name="label").value_counts().sort_index().to_dict()
    )
    test_label_counts = (
        pd.Series(label_encoder.inverse_transform(y_test), name="label").value_counts().sort_index().to_dict()
    )
    total_label_counts = pd.Series(y_text, name="label").value_counts().sort_index().to_dict()

    best_model_name = ""
    best_model = None
    best_model_type = ""
    best_test_input: pd.DataFrame | np.ndarray | None = None
    best_f1 = -1.0
    metrics_rows: list[dict[str, float | str]] = []
    model_registry: dict[str, dict[str, str]] = {}

    for model_name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        model_path = candidate_model_path(model_name=model_name, model_type="sklearn")
        joblib.dump(pipeline, model_path)
        model_registry[model_name] = {"model_type": "sklearn", "path": str(model_path)}

        metrics_rows.append(
            {
                "model": model_name,
                "model_type": "sklearn",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

        if macro_f1 > best_f1:
            best_model_name = model_name
            best_model = pipeline
            best_model_type = "sklearn"
            best_test_input = X_test
            best_f1 = macro_f1

    try:
        from tensorflow import keras

        X_mel = build_mel_dataset(entries)
        X_train_mel = X_mel[train_indices]
        X_test_mel = X_mel[test_indices]

        cnn_model = build_cnn_model(input_shape=X_train_mel.shape[1:], num_classes=len(label_encoder.classes_))
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]
        cnn_model.fit(
            X_train_mel,
            y_train,
            validation_split=0.2,
            epochs=25,
            batch_size=16,
            callbacks=callbacks,
            verbose=0,
        )

        y_pred_probabilities = cnn_model.predict(X_test_mel, verbose=0)
        y_pred = np.argmax(y_pred_probabilities, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        cnn_candidate_path = candidate_model_path(model_name="MelSpectrogramCNN", model_type="cnn")
        cnn_model.save(cnn_candidate_path)
        model_registry["MelSpectrogramCNN"] = {"model_type": "cnn", "path": str(cnn_candidate_path)}
        metrics_rows.append(
            {
                "model": "MelSpectrogramCNN",
                "model_type": "cnn",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

        if macro_f1 > best_f1:
            best_model_name = "MelSpectrogramCNN"
            best_model = cnn_model
            best_model_type = "cnn"
            best_test_input = X_test_mel
            best_f1 = macro_f1
    except ImportError:
        pass

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")
    if best_test_input is None:
        raise RuntimeError("Could not determine held-out test inputs for the best model.")

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if best_model_type == "sklearn":
        joblib.dump(best_model, BEST_MODEL_PATH)
    elif best_model_type == "cnn":
        best_model.save(BEST_CNN_MODEL_PATH)
    else:
        raise ValueError(f"Unsupported best model type: {best_model_type}")

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(model_registry, MODEL_REGISTRY_PATH)
    joblib.dump(
        {
            "model_type": best_model_type,
            "best_model_name": best_model_name,
            "registry": model_registry,
        },
        MODEL_INFO_PATH,
    )
    joblib.dump(
        {
            "X_test_input": best_test_input,
            "y_test": y_test,
            "best_model_name": best_model_name,
            "best_model_type": best_model_type,
            "class_names": list(label_encoder.classes_),
            "dataset_size": int(len(dataset)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "total_label_counts": total_label_counts,
            "train_label_counts": train_label_counts,
            "test_label_counts": test_label_counts,
        },
        EVALUATION_BUNDLE_PATH,
    )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    return metrics_df


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for training."""
    parser = argparse.ArgumentParser(description="Train baseline models on processed keyboard sounds.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory with processed WAV files.",
    )
    return parser


def main() -> None:
    """Run model training from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    metrics_df = train_and_select_best(processed_dir=args.processed_dir)
    print("Training finished. Model comparison:")
    print(metrics_df.to_string(index=False))
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
