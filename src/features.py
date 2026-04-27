"""Feature extraction utilities for short keyboard sound clips."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src.config import DEFAULT_SAMPLE_RATE


def prepare_audio_for_model(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_duration_seconds: float = 0.5,
) -> np.ndarray:
    """Convert audio to mono, target sample rate, and fixed duration."""
    if audio.size == 0:
        raise ValueError("Audio array is empty.")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    if not np.any(np.abs(audio) > 1e-8):
        raise ValueError("Audio appears to contain only silence.")

    target_length = int(target_duration_seconds * sample_rate)
    if audio.size > target_length:
        audio = audio[:target_length]
    elif audio.size < target_length:
        audio = np.pad(audio, (0, target_length - audio.size))

    return audio.astype(np.float32)


def _summarize_feature(feature_matrix: np.ndarray) -> np.ndarray:
    """Convert frame-wise features into a fixed-length vector."""
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix[np.newaxis, :]

    means = np.mean(feature_matrix, axis=1)
    stds = np.std(feature_matrix, axis=1)
    return np.concatenate([means, stds], axis=0)


def _safe_delta(feature_matrix: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute delta features for short clips without failing on small frame counts."""
    frame_count = feature_matrix.shape[1]
    if frame_count < 3:
        return np.zeros_like(feature_matrix)

    width = min(9, frame_count)
    if width % 2 == 0:
        width -= 1

    if width < 3:
        return np.zeros_like(feature_matrix)

    return librosa.feature.delta(feature_matrix, order=order, width=width, mode="nearest")


def extract_features_from_audio(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Extract a fixed-length feature vector from an audio signal."""
    audio = prepare_audio_for_model(audio=audio, sample_rate=sample_rate, target_sample_rate=DEFAULT_SAMPLE_RATE)
    sample_rate = DEFAULT_SAMPLE_RATE

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    delta_mfcc = _safe_delta(mfcc, order=1)
    delta2_mfcc = _safe_delta(mfcc, order=2)
    rms = librosa.feature.rms(y=audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

    features = [
        _summarize_feature(mfcc),
        _summarize_feature(delta_mfcc),
        _summarize_feature(delta2_mfcc),
        _summarize_feature(rms),
        _summarize_feature(spectral_centroid),
        _summarize_feature(zero_crossing_rate),
    ]
    return np.concatenate(features, axis=0).astype(np.float32)


def extract_mel_spectrogram_from_audio(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Extract a normalized mel-spectrogram tensor for CNN input."""
    audio = prepare_audio_for_model(audio=audio, sample_rate=sample_rate, target_sample_rate=DEFAULT_SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=DEFAULT_SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    return mel_db[..., np.newaxis].astype(np.float32)


def extract_features(file_path: str | Path) -> np.ndarray:
    """Load an audio file and extract a fixed-length feature vector."""
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}. Only .wav is supported.")

    audio, sample_rate = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
    return extract_features_from_audio(audio=audio, sample_rate=sample_rate)


def extract_mel_spectrogram(file_path: str | Path) -> np.ndarray:
    """Load an audio file and extract a mel-spectrogram tensor for CNN input."""
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}. Only .wav is supported.")

    audio, sample_rate = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
    return extract_mel_spectrogram_from_audio(audio=audio, sample_rate=sample_rate)
