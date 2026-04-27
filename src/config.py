"""Project-wide configuration constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_SESSIONS_DIR = DATA_DIR / "test_sessions"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CANDIDATES_DIR = MODELS_DIR / "candidates"

DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_RECORD_SECONDS = 0.4
DEFAULT_ALLOWED_LABELS = ["A", "S", "D", "F", "J", "K", "L", "Space", "Enter"]

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_CNN_MODEL_PATH = MODELS_DIR / "best_cnn_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.joblib"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.joblib"
EVALUATION_BUNDLE_PATH = MODELS_DIR / "evaluation_data.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.csv"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
