import base64
import io
import os
import torch
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ---------------- CONFIG ----------------
API_KEY = "YOUR_SECRET_API_KEY"  # Replace this with your key
MODEL_PATH = "voice_detector_model"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
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
    # Convert bytes to numpy array
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Classification
    if probs[1] > probs[0]:
        classification = "AI_GENERATED"
        explanation = "Unnatural pitch consistency and robotic speech patterns detected"
        confidence = float(probs[1])
    else:
        classification = "HUMAN"
        explanation = "Natural voice patterns with normal variations detected"
        confidence = float(probs[0])
    
    return classification, confidence, explanation

# ---------------- API ROUTE ----------------
@app.post("/api/voice-detection")
async def voice_detection(request: Request, x_api_key: str = Header(None)):
    # 1. Validate API Key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key or malformed request")
    
    # 2. Parse JSON body
    data = await request.json()
    language = data.get("language")
    audio_format = data.get("audioFormat")
    audio_base64 = data.get("audioBase64")

    # 3. Validate input
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    if audio_format != "mp3":
        raise HTTPException(status_code=400, detail="audioFormat must be 'mp3'")
    if not audio_base64:
        raise HTTPException(status_code=400, detail="audioBase64 is required")
    
    # 4. Decode Base64
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 5. Predict
    classification, confidence, explanation = predict(audio_bytes)

    # 6. Return JSON
    return {
        "status": "success",
        "language": language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
