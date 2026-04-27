"""Helpers for loading and using saved model artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_CNN_MODEL_PATH,
    BEST_MODEL_PATH,
    LABEL_ENCODER_PATH,
    MODEL_CANDIDATES_DIR,
    MODEL_INFO_PATH,
    MODEL_REGISTRY_PATH,
)


def _resolve_artifact_path(raw_path: str | Path) -> Path:
    """Resolve artifact paths saved on another machine or OS."""
    path = Path(raw_path)
    if path.exists():
        return path

    candidate_names = [path.name]
    candidate_parent = path.parent.name
    if candidate_parent:
        candidate_names.append(f"{candidate_parent}/{path.name}")

    for candidate_name in candidate_names:
        normalized = candidate_name.replace("\\", "/")
        if normalized.startswith("candidates/"):
            candidate_path = MODEL_CANDIDATES_DIR / normalized.split("/", maxsplit=1)[1]
        else:
            candidate_path = BEST_MODEL_PATH.parent / normalized
        if candidate_path.exists():
            return candidate_path

    return path


def _build_fallback_registry() -> dict[str, dict[str, str]]:
    """Recover a minimal registry when metadata is incomplete."""
    registry: dict[str, dict[str, str]] = {}
    if BEST_MODEL_PATH.exists():
        registry["BestModel"] = {"model_type": "sklearn", "path": str(BEST_MODEL_PATH)}
    if BEST_CNN_MODEL_PATH.exists():
        registry["BestCNNModel"] = {"model_type": "cnn", "path": str(BEST_CNN_MODEL_PATH)}
    return registry


def load_model_info() -> dict[str, object]:
    """Load model metadata, with backward-compatible defaults for sklearn-only artifacts."""
    if MODEL_INFO_PATH.exists():
        model_info = dict(joblib.load(MODEL_INFO_PATH))
        registry = dict(model_info.get("registry", {}))
        if not registry and MODEL_REGISTRY_PATH.exists():
            registry = dict(joblib.load(MODEL_REGISTRY_PATH))
        if not registry:
            registry = _build_fallback_registry()
        if registry:
            model_info["registry"] = registry
            if "best_model_name" not in model_info:
                model_info["best_model_name"] = next(iter(registry))
        return model_info

    if BEST_MODEL_PATH.exists():
        return {
            "model_type": "sklearn",
            "best_model_name": "BestModel",
            "registry": _build_fallback_registry(),
        }

    if BEST_CNN_MODEL_PATH.exists():
        return {
            "model_type": "cnn",
            "best_model_name": "BestCNNModel",
            "registry": _build_fallback_registry(),
        }

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
    model_path = _resolve_artifact_path(selected_info["path"])
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
