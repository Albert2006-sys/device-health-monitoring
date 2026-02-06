import librosa
from utils.feature_extractor import extract_features_from_file

# Change this filename if needed
audio_path = "backend/data/train/normal/file_example_WAV_2MG.wav"

features = extract_features_from_file(audio_path)

print("Feature vector length:", len(features))
print("Features:", features)
