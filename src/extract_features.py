import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = "E:\music genre classification\data\genres_original"
GENRES = os.listdir(DATA_PATH)

def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)
    
    # Feature 1: MFCC (13 Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Feature 2: Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Feature 3: Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    # Combine all features into one array
    return np.hstack((mfccs_mean, chroma_mean, contrast_mean))

def create_dataset():
    features = []
    labels = []

    for genre in GENRES:
        genre_path = os.path.join(DATA_PATH, genre)
        for filename in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_path, filename)
                try:
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv("features/genre_features.csv", index=False)
    print("✅ Features saved to 'features/genre_features.csv'")

if __name__ == "__main__":
    os.makedirs("features", exist_ok=True)
    create_dataset()
