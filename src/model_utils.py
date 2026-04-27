"""Helpers for loading and using saved model artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_CNN_MODEL_PATH, BEST_MODEL_PATH, LABEL_ENCODER_PATH, MODEL_INFO_PATH


def load_model_info() -> dict[str, object]:
    """Load model metadata, with backward-compatible defaults for sklearn-only artifacts."""
    if MODEL_INFO_PATH.exists():
        return joblib.load(MODEL_INFO_PATH)

    if BEST_MODEL_PATH.exists():
        return {"model_type": "sklearn", "best_model_name": "UnknownSklearnModel"}

    raise FileNotFoundError("No saved model metadata found. Train the model first.")


def load_prediction_artifacts(model_name: str | None = None) -> tuple[object, object, dict[str, object]]:
    """Load the saved best model or a selected candidate, plus label encoder and metadata."""
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}. Train the model first.")

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model_info = load_model_info()
    registry = dict(model_info.get("registry", {}))

    selected_model_name = model_name or str(model_info["best_model_name"])
    if selected_model_name not in registry:
        available_names = ", ".join(sorted(registry)) if registry else "none"
        raise ValueError(f"Unknown model name: {selected_model_name}. Available models: {available_names}")

    selected_info = dict(registry[selected_model_name])
    model_type = str(selected_info["model_type"])
    model_path = Path(selected_info["path"])
    selected_model_info = dict(model_info)
    selected_model_info["model_type"] = model_type
    selected_model_info["best_model_name"] = selected_model_name
    selected_model_info["selected_model_name"] = selected_model_name
    selected_model_info["selected_model_type"] = model_type

    if model_type == "sklearn":
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")
        model = joblib.load(model_path)
        return model, label_encoder, selected_model_info

    if model_type == "cnn":
        if not model_path.exists():
            raise FileNotFoundError(f"CNN model file not found: {model_path}. Train the model first.")
        try:
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
            raise ImportError(
                "TensorFlow is required to load the saved CNN model. Install `tensorflow-cpu`."
            ) from exc

        model = keras.models.load_model(model_path)
        return model, label_encoder, selected_model_info

    raise ValueError(f"Unsupported model type in metadata: {model_type}")


def predict_probabilities(model: object, model_type: str, model_input: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Run probability prediction for sklearn or CNN models."""
    if model_type == "cnn":
        probabilities = model.predict(model_input, verbose=0)
        return np.asarray(probabilities, dtype=np.float32)

    probabilities = model.predict_proba(model_input)
    return np.asarray(probabilities, dtype=np.float32)
