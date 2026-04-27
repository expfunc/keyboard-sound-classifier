"""Predict a keyboard key label from one WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import extract_features, extract_features_from_audio, extract_mel_spectrogram_from_audio
from src.model_utils import load_prediction_artifacts, predict_probabilities


def predict_feature_vector(
    feature_vector: np.ndarray,
    model: object,
    label_encoder: object,
    model_info: dict[str, object],
) -> dict[str, object]:
    """Predict a label from one already-extracted feature vector."""
    model_type = str(model_info["model_type"])
    if model_type == "cnn":
        raise ValueError("CNN models require mel-spectrogram input, not tabular feature vectors.")

    model_input: np.ndarray | pd.DataFrame = feature_vector
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        model_input = pd.DataFrame(feature_vector, columns=list(feature_names))
    encoded_prediction = model.predict(model_input)[0]
    predicted_label = label_encoder.inverse_transform([encoded_prediction])[0]

    result: dict[str, object] = {"label": predicted_label}
    if hasattr(model, "predict_proba"):
        probabilities = predict_probabilities(model=model, model_type=model_type, model_input=model_input)[0]
        class_probabilities = {
            label: float(probability)
            for label, probability in zip(label_encoder.classes_, probabilities, strict=True)
        }
        result["probabilities"] = class_probabilities
        result["confidence"] = max(class_probabilities.values())

    return result


def predict_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    model: object,
    label_encoder: object,
    model_info: dict[str, object],
) -> dict[str, object]:
    """Predict a label from an in-memory audio array."""
    model_type = str(model_info["model_type"])

    if model_type == "cnn":
        mel_tensor = extract_mel_spectrogram_from_audio(audio=audio, sample_rate=sample_rate)[np.newaxis, ...]
        probabilities = predict_probabilities(model=model, model_type=model_type, model_input=mel_tensor)[0]
        encoded_prediction = int(np.argmax(probabilities))
        predicted_label = label_encoder.inverse_transform([encoded_prediction])[0]
        class_probabilities = {
            label: float(probability)
            for label, probability in zip(label_encoder.classes_, probabilities, strict=True)
        }
        return {
            "label": predicted_label,
            "probabilities": class_probabilities,
            "confidence": max(class_probabilities.values()),
        }

    feature_vector = extract_features_from_audio(audio=audio, sample_rate=sample_rate).reshape(1, -1)
    return predict_feature_vector(
        feature_vector=feature_vector,
        model=model,
        label_encoder=label_encoder,
        model_info=model_info,
    )


def predict_file(audio_path: str | Path) -> dict[str, object]:
    """Load the saved model and predict the label for a single WAV file."""
    file_path = Path(audio_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model, label_encoder, model_info = load_prediction_artifacts()
    if str(model_info["model_type"]) == "cnn":
        import librosa

        audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        return predict_audio_array(
            audio=audio,
            sample_rate=sample_rate,
            model=model,
            label_encoder=label_encoder,
            model_info=model_info,
        )

    feature_vector = extract_features(file_path).reshape(1, -1)
    return predict_feature_vector(
        feature_vector=feature_vector,
        model=model,
        label_encoder=label_encoder,
        model_info=model_info,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for single-file prediction."""
    parser = argparse.ArgumentParser(description="Predict a key label from a WAV file.")
    parser.add_argument("audio_path", type=Path, help="Path to a WAV file.")
    return parser


def main() -> None:
    """Run single-file prediction from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    result = predict_file(args.audio_path)
    print(f"Predicted label: {result['label']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "probabilities" in result:
        print("Class probabilities:")
        for label, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
