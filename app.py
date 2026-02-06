import base64
import io
import os
import torch
import librosa
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from google.cloud import storage

# ---------------- CONFIG ----------------
API_KEY = "YOUR_SECRET_API_KEY"  # change this to your key

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

MODEL_DIR = "voice_detector_model"

# ✅ YOUR BUCKET

GCS_BUCKET_NAME = "voice-ai-models-haany"
GCS_MODEL_PREFIX = "voice_detector_model/"  # folder name inside bucket

# ---------------- DOWNLOAD MODEL FROM GCS ----------------
def download_model_from_gcs():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GCS_MODEL_PREFIX)

    os.makedirs(MODEL_DIR, exist_ok=True)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        local_path = os.path.join(MODEL_DIR, blob.name.replace(GCS_MODEL_PREFIX, ""))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

# ---------------- LOAD MODEL SAFELY ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print("Downloading model from GCS...")
    download_model_from_gcs()
    print("Model downloaded.")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

model.to(device)
model.eval()

# ---------------- FASTAPI ----------------
app = FastAPI(title="AI Voice Detection API")

# ---------------- REQUEST BODY ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ---------------- HELPER FUNCTION ----------------
def predict(audio_bytes):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    if probs[1] > probs[0]:
        return "AI_GENERATED", float(probs[1]), "Unnatural pitch consistency and robotic speech patterns detected"
    else:
        return "HUMAN", float(probs[0]), "Natural voice patterns with normal variations detected"

# ---------------- API ROUTE ----------------
@app.post("/api/voice-detection")
def voice_detection(data: VoiceRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key or malformed request")

    if data.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {data.language}")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="audioFormat must be 'mp3'")

    if not data.audioBase64:
        raise HTTPException(status_code=400, detail="audioBase64 is required")

    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    classification, confidence, explanation = predict(audio_bytes)

    return {
        "status": "success",
        "language": data.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

