"""Streamlit app for the keyboard sound classifier MVP."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from src.model_utils import load_prediction_artifacts
from src.predict import predict_audio_array


st.set_page_config(page_title="Keyboard Sound Classifier", page_icon="⌨️")
st.title("Keyboard Sound Classifier")
st.caption("Educational MVP for classifying short keyboard click recordings that you recorded yourself.")

try:
    model, label_encoder, model_info = load_prediction_artifacts()
except Exception as exc:
    st.error(f"Model artifacts were not found or could not be loaded: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        audio, sample_rate = sf.read(io.BytesIO(uploaded_file.getvalue()), dtype="float32")
    except Exception as exc:
        st.error(f"Could not read uploaded audio: {exc}")
        st.stop()

    if np.size(audio) == 0:
        st.error("Uploaded file is empty.")
        st.stop()

    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)

    try:
        prediction = predict_audio_array(
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
    st.write(f"Predicted key: **{prediction['label']}**")

    if "probabilities" in prediction:
        probability_table = prediction["probabilities"]
        st.write(f"Confidence: **{float(prediction['confidence']):.4f}**")
        st.bar_chart(pd.Series(probability_table))
    else:
        st.info("This model does not expose class probabilities.")

st.divider()
st.write(
    "Use this app only with your own explicit recordings for educational audio classification. "
    "It is not designed for background recording, password capture, or covert monitoring."
)
