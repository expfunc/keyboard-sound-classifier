"""Interactive session recorder for a keyboard sound dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from src.config import DEFAULT_ALLOWED_LABELS, DEFAULT_RECORD_SECONDS, DEFAULT_SAMPLE_RATE, RAW_DATA_DIR


def choose_label(allowed_labels: list[str]) -> str:
    """Prompt the user to choose a label from the allowed list."""
    print("Available labels:")
    for index, label in enumerate(allowed_labels, start=1):
        print(f"{index}. {label}")

    selection = input("Choose a label by name or number: ").strip()
    if selection.isdigit():
        label_index = int(selection) - 1
        if 0 <= label_index < len(allowed_labels):
            return allowed_labels[label_index]
    if selection in allowed_labels:
        return selection
    raise ValueError(f"Invalid label: {selection}")


def list_input_devices() -> list[tuple[int, dict]]:
    """Return all available input devices."""
    devices = sd.query_devices()
    input_devices = [
        (index, device_info)
        for index, device_info in enumerate(devices)
        if int(device_info["max_input_channels"]) > 0
    ]
    if not input_devices:
        raise RuntimeError("No input audio devices were found.")
    return input_devices


def choose_input_device() -> int:
    """Prompt the user to choose a microphone/input device."""
    input_devices = list_input_devices()
    print("Available microphones:")
    for display_index, (device_index, device_info) in enumerate(input_devices, start=1):
        default_sample_rate = int(device_info["default_samplerate"])
        print(f"{display_index}. [{device_index}] {device_info['name']} (default {default_sample_rate} Hz)")

    selection = input("Choose a microphone by number or device index: ").strip()
    if selection.isdigit():
        numeric_value = int(selection)
        if 1 <= numeric_value <= len(input_devices):
            return input_devices[numeric_value - 1][0]
        valid_indices = {device_index for device_index, _ in input_devices}
        if numeric_value in valid_indices:
            return numeric_value

    raise ValueError(f"Invalid microphone selection: {selection}")


def next_file_index(label_dir: Path, label: str) -> int:
    """Return the next numeric file index for a label directory."""
    existing_files = sorted(label_dir.glob(f"{label}_*.wav"))
    if not existing_files:
        return 1

    existing_indices: list[int] = []
    for file_path in existing_files:
        try:
            existing_indices.append(int(file_path.stem.rsplit("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue

    return max(existing_indices, default=0) + 1


def detect_click_onsets(audio: np.ndarray, sample_rate: int, min_separation_seconds: float = 0.12) -> np.ndarray:
    """Detect likely key-click onsets in a recording."""
    if audio.size == 0:
        return np.array([], dtype=np.int64)

    normalized = audio.astype(np.float32)
    peak = float(np.max(np.abs(normalized)))
    if peak > 1e-8:
        normalized = normalized / peak

    hop_length = 128
    onset_envelope = librosa.onset.onset_strength(y=normalized, sr=sample_rate, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.2,
        wait=max(1, int(min_separation_seconds * sample_rate / hop_length)),
    )
    return librosa.frames_to_samples(onset_frames, hop_length=hop_length)


def record_session_until_clicks_detected(
    target_clicks: int,
    sample_rate: int,
    device: int,
    max_duration_seconds: float,
    tail_padding_seconds: float = 0.3,
    block_size: int = 2048,
) -> np.ndarray:
    """Record a continuous session and stop after enough clicks are detected."""
    sd.check_input_settings(device=device, samplerate=sample_rate, channels=1)

    buffers: list[np.ndarray] = []
    detected_clicks = 0
    tail_blocks_remaining: int | None = None
    total_frames_limit = int(max_duration_seconds * sample_rate)

    print("Recording started. Press the selected key repeatedly.")
    print(f"The recorder will stop automatically after it detects {target_clicks} clicks.")

    with sd.InputStream(
        samplerate=sample_rate,
        device=device,
        channels=1,
        dtype="float32",
        blocksize=block_size,
    ) as stream:
        while True:
            chunk, overflowed = stream.read(block_size)
            if overflowed:
                print("Warning: audio buffer overflow detected. Consider recording in a quieter environment.")

            buffers.append(chunk[:, 0].copy())
            audio = np.concatenate(buffers)

            current_onsets = detect_click_onsets(audio, sample_rate=sample_rate)
            current_count = len(current_onsets)
            if current_count != detected_clicks:
                detected_clicks = current_count
                print(f"Detected clicks: {min(detected_clicks, target_clicks)}/{target_clicks}")

            if detected_clicks >= target_clicks and tail_blocks_remaining is None:
                tail_blocks_remaining = max(1, int(np.ceil(tail_padding_seconds * sample_rate / block_size)))
            elif tail_blocks_remaining is not None:
                tail_blocks_remaining -= 1
                if tail_blocks_remaining <= 0:
                    break

            if audio.size >= total_frames_limit:
                break

    session_audio = np.concatenate(buffers) if buffers else np.array([], dtype=np.float32)
    if session_audio.size == 0:
        raise RuntimeError("Recording failed: no audio frames were captured.")

    return session_audio


def split_click_recording(
    audio: np.ndarray,
    sample_rate: int,
    target_clicks: int,
    clip_duration: float,
    pre_click_seconds: float = 0.05,
) -> list[np.ndarray]:
    """Split a continuous session into fixed-length click clips."""
    onsets = detect_click_onsets(audio, sample_rate=sample_rate)
    if len(onsets) < target_clicks:
        raise ValueError(
            f"Only {len(onsets)} click(s) were detected, but {target_clicks} were requested. "
            "Try recording again with clearer gaps between clicks."
        )

    clip_samples = int(clip_duration * sample_rate)
    pre_click_samples = int(pre_click_seconds * sample_rate)
    segments: list[np.ndarray] = []

    for onset in onsets[:target_clicks]:
        start = max(0, int(onset) - pre_click_samples)
        end = min(audio.size, start + clip_samples)
        segment = np.zeros(clip_samples, dtype=np.float32)
        extracted = audio[start:end]
        segment[: extracted.size] = extracted
        segments.append(segment)

    return segments


def save_recording(audio: np.ndarray, sample_rate: int, output_path: Path) -> None:
    """Save a recorded clip as a WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for dataset recording."""
    parser = argparse.ArgumentParser(description="Record a continuous keyboard-click session for your own dataset.")
    parser.add_argument("--label", type=str, help="Keyboard label to record, for example A or Space.")
    parser.add_argument("--samples", type=int, default=10, help="Number of clips to record.")
    parser.add_argument("--duration", type=float, default=DEFAULT_RECORD_SECONDS, help="Clip duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Recording sample rate.")
    parser.add_argument("--device", type=int, help="Optional input device index. If omitted, you will choose one.")
    parser.add_argument(
        "--max-session-seconds",
        type=float,
        default=30.0,
        help="Maximum length of the continuous recording session before auto-stop.",
    )
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR, help="Root directory for raw audio.")
    return parser


def main() -> None:
    """Run the interactive recording workflow."""
    parser = build_parser()
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero.")
    if not 0.3 <= args.duration <= 0.5:
        raise ValueError("--duration must be between 0.3 and 0.5 seconds for this MVP.")
    if args.max_session_seconds <= 0:
        raise ValueError("--max-session-seconds must be greater than zero.")

    label = args.label or choose_label(DEFAULT_ALLOWED_LABELS)
    if label not in DEFAULT_ALLOWED_LABELS:
        raise ValueError(f"Label must be one of: {', '.join(DEFAULT_ALLOWED_LABELS)}")

    device = args.device if args.device is not None else choose_input_device()
    label_dir = args.output_dir / label
    start_index = next_file_index(label_dir, label)

    print("This recorder is intended only for your own, explicit, foreground recordings.")
    print("You will press Enter once, then repeatedly tap the same key.")
    print("The script will detect individual clicks and save them as separate WAV files.")
    input(f"Press Enter to start the recording session for label {label}...")

    session_audio = record_session_until_clicks_detected(
        target_clicks=args.samples,
        sample_rate=args.sample_rate,
        device=device,
        max_duration_seconds=args.max_session_seconds,
    )
    click_segments = split_click_recording(
        audio=session_audio,
        sample_rate=args.sample_rate,
        target_clicks=args.samples,
        clip_duration=args.duration,
    )

    for offset, segment in enumerate(click_segments):
        file_index = start_index + offset
        output_path = label_dir / f"{label}_{file_index:03d}.wav"
        save_recording(audio=segment, sample_rate=args.sample_rate, output_path=output_path)
        print(f"Saved: {output_path}")

    print(f"Finished. Saved {len(click_segments)} clip(s) for label {label}.")


if __name__ == "__main__":
    main()
