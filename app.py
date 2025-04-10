import streamlit as st
import librosa
import numpy as np
import joblib
import os

# Load model and preprocessing tools
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    return np.hstack((mfccs_mean, chroma_mean, contrast_mean))

# Streamlit UI
st.title("🎵 Music Genre Classifier")
st.write("Upload a 30-sec `.wav` audio file to predict its genre.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.audio("temp.wav")

    if st.button("Predict Genre"):
        try:
            features = extract_features("temp.wav")
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            genre = label_encoder.inverse_transform(prediction)
            st.success(f"🎧 Predicted Genre: **{genre[0]}**")
        except Exception as e:
            st.error(f"Error: {e}")
