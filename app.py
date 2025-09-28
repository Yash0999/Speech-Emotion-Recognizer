import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging
from flask_cors import CORS
import traceback
import sys
import soundfile as sf

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Emotion labels and emojis
emotions = ["angry", "disgust", "fear", "neutral", "sad"]
emojis = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "neutral": "😐",
    "sad": "😢",
}

# Feature extraction function with detailed logging
def extract_features(file_path):
    try:
        logger.info(f"Extracting features from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Check if file is empty
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        # Load audio file with multiple fallback options
        y = None
        sr = None
        
        # Try with librosa first
        try:
            y, sr = librosa.load(file_path, sr=None)
            logger.info(f"Audio loaded with librosa: sample rate={sr}, duration={len(y)/sr:.2f}s, shape={y.shape}")
        except Exception as e:
            logger.warning(f"librosa failed to load file: {str(e)}")
            
            # Try with soundfile as fallback
            try:
                y, sr = sf.read(file_path)
                logger.info(f"Audio loaded with soundfile: sample rate={sr}, duration={len(y)/sr:.2f}s, shape={y.shape}")
                
                # Convert to mono if stereo
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = np.mean(y, axis=1)
                    logger.info(f"Converted to mono: shape={y.shape}")
            except Exception as e2:
                logger.error(f"Both librosa and soundfile failed: {str(e2)}")
                raise ValueError(f"Could not load audio file: {str(e2)}")
        
        # Check if audio is empty
        if len(y) == 0:
            raise ValueError("Audio file is empty after loading")
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Log feature shapes
        logger.info(f"MFCC shape: {mfccs.shape}")
        logger.info(f"Chroma shape: {chroma.shape}")
        logger.info(f"Mel shape: {mel.shape}")
        
        # Compute mean of features
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel_mean = np.mean(mel.T, axis=0)
        
        # Log mean feature shapes
        logger.info(f"MFCC mean shape: {mfccs_mean.shape}")
        logger.info(f"Chroma mean shape: {chroma_mean.shape}")
        logger.info(f"Mel mean shape: {mel_mean.shape}")
        
        # Concatenate features
        features = np.concatenate([mfccs_mean, chroma_mean, mel_mean])
        logger.info(f"Final features shape: {features.shape}")
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# SERModel definition
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

# Load the trained model with error handling
try:
    input_size = 180  # Ensure this matches your training input
    num_classes = len(emotions)
    model = SERModel(input_size=input_size, num_classes=num_classes)
    
    # Check if model file exists
    model_path = 'best_ser_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Prediction function with detailed logging
def predict_emotion_with_probabilities(file_path):
    try:
        logger.info(f"Starting prediction for {file_path}")
        features = extract_features(file_path)
        
        # Check feature size
        if features.shape[0] != input_size:
            error_msg = f"Feature size mismatch: expected {input_size}, got {features.shape[0]}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        logger.info(f"Features tensor shape: {features.shape}")
        
        with torch.no_grad():
            outputs = model(features)
            logger.info(f"Model outputs: {outputs}")
            probabilities = torch.softmax(outputs, dim=1)
            logger.info(f"Probabilities: {probabilities}")
            top2_probabilities, top2_indices = torch.topk(probabilities, 2)
            logger.info(f"Top 2 indices: {top2_indices}, probabilities: {top2_probabilities}")
        
        top2_emotions = [emotions[idx] for idx in top2_indices[0]]
        top2_probs = top2_probabilities[0].numpy()
        
        result = [{"emotion": top2_emotions[i], "probability": float(top2_probs[i]), "emoji": emojis[top2_emotions[i]]} for i in range(2)]
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# API route to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    filepath = None  # Initialize to None for the finally block
    try:
        logger.info("Received prediction request")
        
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({"error": "No file selected."}), 400

        # Log file details
        logger.info(f"File received: {file.filename}, content type: {file.content_type}")

        # Save and process the uploaded file
        filename = secure_filename(file.filename)
        # Ensure the file has a .wav extension
        if not filename.lower().endswith(('.wav', '.mp3', '.ogg', '.webm')):
            filename += '.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to {filepath}")

        # Verify file was saved
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File was not saved successfully: {filepath}")

        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise ValueError("File is empty")

        # Predict emotions
        predictions = predict_emotion_with_probabilities(filepath)
        return jsonify({"predictions": predictions})
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Temporary file {filepath} removed")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

# Main route for the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER'])
    })

# Test endpoint for file processing
@app.route('/test-file', methods=['POST'])
def test_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.wav', '.mp3', '.ogg', '.webm')):
            filename += '.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get file info
        file_size = os.path.getsize(filepath)
        
        # Try to load with librosa
        try:
            y, sr = librosa.load(filepath, sr=None)
            duration = len(y) / sr
            audio_shape = y.shape
            load_method = "librosa"
        except:
            # Try with soundfile as fallback
            y, sr = sf.read(filepath)
            duration = len(y) / sr
            audio_shape = y.shape
            load_method = "soundfile"
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            "filename": filename,
            "file_size": file_size,
            "sample_rate": sr,
            "duration": duration,
            "audio_shape": audio_shape,
            "load_method": load_method
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)