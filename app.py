import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging
from flask_cors import CORS  # CORS import moved here

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS after the app is initialized

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels and emojis
emotions = ["angry", "disgust", "fear", "neutral", "sad"]
emojis = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "neutral": "😐",
    "sad": "😢",
}

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features = np.concatenate(
        [np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0)]
    )
    return features

# SERModel definition (ensure this matches the training model structure)
class SERModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SERModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x

# Load the trained model
input_size = 180  # Ensure this matches your training input
num_classes = len(emotions)
model = SERModel(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load('best_ser_model.pth', map_location=torch.device('cpu')))
model.eval()

# Prediction function
def predict_emotion_with_probabilities(file_path):
    features = extract_features(file_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        top2_probabilities, top2_indices = torch.topk(probabilities, 2)
    top2_emotions = [emotions[idx] for idx in top2_indices[0]]
    top2_probs = top2_probabilities[0].numpy()
    # Convert probabilities to Python float for JSON serialization
    return [{"emotion": top2_emotions[i], "probability": float(top2_probs[i]), "emoji": emojis[top2_emotions[i]]} for i in range(2)]

# API route to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Save and process the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict emotions
    try:
        predictions = predict_emotion_with_probabilities(filepath)
        return jsonify({"predictions": predictions})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# Main route for the frontend
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
