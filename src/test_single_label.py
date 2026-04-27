"""Safe foreground test mode for one expected keyboard label."""

from __future__ import annotations

import argparse
import msvcrt
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.config import (
    DEFAULT_ALLOWED_LABELS,
    DEFAULT_RECORD_SECONDS,
    DEFAULT_SAMPLE_RATE,
    TEST_SESSIONS_DIR,
)
from src.model_utils import load_prediction_artifacts
from src.predict import predict_audio_array
from src.record import choose_input_device, detect_click_onsets, split_click_recording


def wait_for_end_key(stop_event: threading.Event) -> None:
    """Stop recording after the user presses End in the active console."""
    while not stop_event.is_set():
        if not msvcrt.kbhit():
            time.sleep(0.05)
            continue

        first_char = msvcrt.getwch()
        if first_char not in ("\x00", "\xe0"):
            continue

        second_char = msvcrt.getwch()
        if second_char == "O":
            stop_event.set()
            return


def wait_for_stop_command(stop_event: threading.Event) -> None:
    """Stop recording after the user types /stop and presses Enter."""
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            return

        if line.strip().lower() == "/stop":
            stop_event.set()
            return


def record_until_end_key(sample_rate: int, device: int, max_duration_seconds: float, block_size: int = 2048) -> np.ndarray:
    """Record audio until the user presses End or enters /stop."""
    sd.check_input_settings(device=device, samplerate=sample_rate, channels=1)

    stop_event = threading.Event()
    end_key_thread = threading.Thread(target=wait_for_end_key, args=(stop_event,), daemon=True)
    stop_command_thread = threading.Thread(target=wait_for_stop_command, args=(stop_event,), daemon=True)
    end_key_thread.start()
    stop_command_thread.start()

    print("Recording started.")
    print("Press the same key repeatedly for the test.")
    print("When finished, press End in this console.")
    print("If End is intercepted by the terminal, type /stop and press Enter.")

    buffers: list[np.ndarray] = []
    total_frames_limit = int(max_duration_seconds * sample_rate)

    with sd.InputStream(
        samplerate=sample_rate,
        device=device,
        channels=1,
        dtype="float32",
        blocksize=block_size,
    ) as stream:
        while not stop_event.is_set():
            chunk, overflowed = stream.read(block_size)
            if overflowed:
                print("Warning: audio buffer overflow detected.")
            buffers.append(chunk[:, 0].copy())

            current_frames = sum(buffer.size for buffer in buffers)
            if current_frames >= total_frames_limit:
                print("Stopped because max session duration was reached.")
                break

    audio = np.concatenate(buffers) if buffers else np.array([], dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Recording failed: no audio frames were captured.")
    return audio


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for the safe test mode."""
    parser = argparse.ArgumentParser(
        description="Foreground test mode for repeated presses of one keyboard label.",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Optional expected keyboard label. If set, the script will also calculate session accuracy.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Optional saved model name to use, for example RandomForest, SVC, KNeighbors, or MelSpectrogramCNN.",
    )
    parser.add_argument("--device", type=int, help="Optional input device index. If omitted, you will choose one.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Recording sample rate.")
    parser.add_argument("--duration", type=float, default=DEFAULT_RECORD_SECONDS, help="Per-click clip duration.")
    parser.add_argument(
        "--max-session-seconds",
        type=float,
        default=20.0,
        help="Maximum recording time before auto-stop.",
    )
    parser.add_argument(
        "--save-session",
        action="store_true",
        help="Save the full test session WAV into data/test_sessions/ for manual inspection.",
    )
    return parser


def main() -> None:
    """Run the safe single-label test workflow."""
    parser = build_parser()
    args = parser.parse_args()

    if not 0.3 <= args.duration <= 0.5:
        raise ValueError("--duration must be between 0.3 and 0.5 seconds for this MVP.")
    if args.max_session_seconds <= 0:
        raise ValueError("--max-session-seconds must be greater than zero.")

    expected_label = args.label
    if expected_label is not None and expected_label not in DEFAULT_ALLOWED_LABELS:
        raise ValueError(f"Label must be one of: {', '.join(DEFAULT_ALLOWED_LABELS)}")

    device = args.device if args.device is not None else choose_input_device()
    model, label_encoder, model_info = load_prediction_artifacts(model_name=args.model_name)

    print("This test mode records an explicit foreground session and prints model predictions for each detected click.")
    print(
        f"Using model: {model_info['selected_model_name']} "
        f"({model_info['selected_model_type']})"
    )
    if expected_label is not None:
        print(f"Expected label for this session: {expected_label}")
    input("Press Enter to start the test session...")

    session_audio = record_until_end_key(
        sample_rate=args.sample_rate,
        device=device,
        max_duration_seconds=args.max_session_seconds,
    )

    if args.save_session:
        TEST_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        label_part = expected_label if expected_label is not None else "unlabeled"
        session_path = TEST_SESSIONS_DIR / f"test_{label_part}_{timestamp}.wav"
        sf.write(session_path, session_audio, args.sample_rate)
        print(f"Saved session audio: {session_path}")

    detected_clicks = len(detect_click_onsets(session_audio, sample_rate=args.sample_rate))
    if detected_clicks == 0:
        raise RuntimeError("No clicks were detected in the test recording.")

    all_segments = split_click_recording(
        audio=session_audio,
        sample_rate=args.sample_rate,
        target_clicks=detected_clicks,
        clip_duration=args.duration,
    )

    predictions: list[dict[str, object]] = []
    for segment in all_segments:
        prediction = predict_audio_array(
            segment,
            sample_rate=args.sample_rate,
            model=model,
            label_encoder=label_encoder,
            model_info=model_info,
        )
        predictions.append(prediction)

    print(f"Detected clicks: {len(predictions)}")
    correct_predictions = 0
    label_counts: dict[str, int] = {}
    predicted_sequence: list[str] = []

    for index, prediction in enumerate(predictions, start=1):
        predicted_label = str(prediction["label"])
        predicted_sequence.append(predicted_label)
        label_counts[predicted_label] = label_counts.get(predicted_label, 0) + 1
        confidence_text = ""
        if "confidence" in prediction:
            confidence_text = f", confidence={float(prediction['confidence']):.4f}"
        print(f"{index:02d}. predicted={predicted_label}{confidence_text}")
        if expected_label is not None and predicted_label == expected_label:
            correct_predictions += 1

    print(f"Predicted sequence: {' '.join(predicted_sequence)}")
    if expected_label is not None:
        accuracy = correct_predictions / len(predictions)
        print(f"Expected label: {expected_label}")
        print(f"Session accuracy for expected label: {accuracy:.4f}")
    print("Prediction counts:")
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
