import os
import librosa
import numpy as np
import joblib

# Load saved model and objects
model = joblib.load("models/random_forest.pkl")  # You can switch to svm_model.pkl or knn_model.pkl
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    return np.hstack((mfccs_mean, chroma_mean, contrast_mean))

def predict_genre(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    genre = label_encoder.inverse_transform(prediction)
    
    print(f"🎵 Predicted Genre: {genre[0]}")

if __name__ == "__main__":
    # Example: predict from a new song
    test_file = "E:\music genre classification\data\genres_original\hiphop\hiphop.00018.wav"  # Replace with your audio file
    predict_genre(test_file)
