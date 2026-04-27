"""Dataset preprocessing utilities and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.config import DEFAULT_SAMPLE_RATE, PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_audio(file_path: str | Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono audio with a target sample rate."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Unsupported audio format: {path.suffix}. Only .wav is supported.")

    try:
        audio, sample_rate = librosa.load(path, sr=target_sample_rate, mono=True)
    except Exception as exc:  # pragma: no cover - depends on backend errors
        raise ValueError(f"Failed to load audio file: {path}") from exc

    return audio, sample_rate


def trim_silence(audio: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio if trimmed_audio.size > 0 else audio


def normalize_audio(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize audio to a safe peak level."""
    peak = float(np.max(np.abs(audio)))
    if peak < eps:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def process_audio_file(source_path: str | Path, destination_path: str | Path) -> Path:
    """Load, trim, normalize, and save a processed audio file."""
    audio, sample_rate = load_audio(source_path)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate)
    return destination


def process_dataset(
    raw_dir: str | Path = RAW_DATA_DIR,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> int:
    """Process every WAV file from the raw dataset directory."""
    raw_root = Path(raw_dir)
    processed_root = Path(processed_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_root}")

    wav_files = sorted(raw_root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in: {raw_root}")

    processed_count = 0
    for source_path in wav_files:
        relative_path = source_path.relative_to(raw_root)
        destination_path = processed_root / relative_path
        process_audio_file(source_path, destination_path)
        processed_count += 1

    return processed_count


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for dataset preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess a keyboard sound dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Directory with raw WAV files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory where processed WAV files will be saved.",
    )
    return parser


def main() -> None:
    """Run dataset preprocessing from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    processed_count = process_dataset(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
    print(f"Processed {processed_count} file(s) into: {args.processed_dir}")


if __name__ == "__main__":
    main()
