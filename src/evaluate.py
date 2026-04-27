"""Evaluate the saved best model on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from src.config import CONFUSION_MATRIX_PATH, EVALUATION_BUNDLE_PATH
from src.model_utils import load_prediction_artifacts


def evaluate_model(
    evaluation_bundle_path: str | Path = EVALUATION_BUNDLE_PATH,
) -> dict[str, object]:
    """Load saved artifacts and compute evaluation metrics."""
    bundle_file = Path(evaluation_bundle_path)

    if not bundle_file.exists():
        raise FileNotFoundError(f"Evaluation data not found: {bundle_file}. Train the model first.")

    model, _label_encoder, model_info = load_prediction_artifacts()
    evaluation_bundle = joblib.load(bundle_file)

    X_test = evaluation_bundle["X_test_input"]
    y_test = evaluation_bundle["y_test"]
    class_names = evaluation_bundle["class_names"]
    model_type = str(evaluation_bundle.get("best_model_type", model_info["model_type"]))

    if model_type == "cnn":
        y_pred_probabilities = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probabilities, axis=1)
    else:
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)

    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        cmap="Blues",
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix_path": CONFUSION_MATRIX_PATH,
        "best_model_name": evaluation_bundle.get("best_model_name"),
        "best_model_type": model_type,
        "dataset_size": evaluation_bundle.get("dataset_size"),
        "train_size": evaluation_bundle.get("train_size"),
        "test_size": evaluation_bundle.get("test_size"),
        "total_label_counts": evaluation_bundle.get("total_label_counts", {}),
        "train_label_counts": evaluation_bundle.get("train_label_counts", {}),
        "test_label_counts": evaluation_bundle.get("test_label_counts", {}),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for evaluation."""
    return argparse.ArgumentParser(description="Evaluate the saved keyboard sound classifier.")


def main() -> None:
    """Run evaluation from the command line."""
    build_parser().parse_args()
    results = evaluate_model()
    print(f"Best model: {results['best_model_name']} ({results['best_model_type']})")
    if results["dataset_size"] is not None:
        print(f"Dataset size: {results['dataset_size']}")
        print(f"Train size: {results['train_size']}")
        print(f"Test size: {results['test_size']}")
        print("Per-class split:")
        total_counts = results["total_label_counts"]
        train_counts = results["train_label_counts"]
        test_counts = results["test_label_counts"]
        for label in sorted(total_counts):
            total_count = total_counts.get(label, 0)
            train_count = train_counts.get(label, 0)
            test_count = test_counts.get(label, 0)
            print(f"  {label}: total={total_count}, train={train_count}, test={test_count}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("Classification report:")
    print(results["report"])
    print(f"Confusion matrix saved to: {results['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
