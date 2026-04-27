"""Streamlit app for the keyboard sound classifier MVP."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from src.model_utils import load_prediction_artifacts
from src.predict import predict_click_sequence


st.set_page_config(page_title="Keyboard Sound Classifier", page_icon="⌨️")
st.title("Keyboard Sound Classifier")
st.caption("Educational MVP for classifying short keyboard click recordings that you recorded yourself.")

try:
    model, label_encoder, model_info = load_prediction_artifacts()
except Exception as exc:
    st.error(f"Model artifacts were not found or could not be loaded: {exc}")
    st.stop()

st.subheader("Audio Source")
source_mode = st.radio(
    "Choose how to provide audio",
    options=["Record from microphone", "Upload WAV file"],
    horizontal=True,
)

audio_source = None
if source_mode == "Record from microphone":
    audio_input = getattr(st, "audio_input", None)
    if audio_input is None:
        st.error("This Streamlit version does not support microphone recording. Use WAV upload instead.")
        st.stop()
    st.caption("Use the microphone widget below to start and stop recording.")
    audio_source = audio_input("Record from your microphone", sample_rate=22_050)
else:
    audio_source = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_source is not None:
    try:
        audio_bytes = audio_source.getvalue()
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:
        st.error(f"Could not read audio: {exc}")
        st.stop()

    if np.size(audio) == 0:
        st.error("Recorded or uploaded audio is empty.")
        st.stop()

    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)

    st.audio(audio_bytes, format="audio/wav")

    try:
        prediction = predict_click_sequence(
            audio=audio,
            sample_rate=sample_rate,
            model=model,
            label_encoder=label_encoder,
            model_info=model_info,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Prediction")
    st.write(f"Detected clicks: **{prediction['detected_clicks']}**")
    st.write(f"Predicted sequence: **{prediction['sequence_text']}**")

    prediction_rows = pd.DataFrame(prediction["predictions"])
    if not prediction_rows.empty:
        display_rows = prediction_rows[["index", "label", "confidence"]].copy()
        display_rows = display_rows.rename(
            columns={"index": "Click", "label": "Predicted key", "confidence": "Confidence"}
        )
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

    st.subheader("Prediction Counts")
    st.bar_chart(pd.Series(prediction["label_counts"]).sort_values(ascending=False))

    first_probabilities = next(
        (
            item["probabilities"]
            for item in prediction["predictions"]
            if item.get("probabilities") is not None
        ),
        None,
    )
    if first_probabilities is not None:
        st.subheader("Class Probabilities For First Detected Click")
        st.bar_chart(pd.Series(first_probabilities))
    else:
        st.info("This model does not expose class probabilities.")

st.divider()
st.write(
    "Use this app only with your own explicit recordings for educational audio classification. "
    "It is not designed for background recording, password capture, or covert monitoring."
)
