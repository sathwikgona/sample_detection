import librosa
import numpy as np
from fastapi import UploadFile
import tempfile

async def predict_audio(file: UploadFile):
    try:
        # Save temp file
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(contents)
            path = f.name

        # Load audio
        y, sr = librosa.load(path, sr=22050)
        energy = np.mean(np.abs(y))

        confidence = float(min(1.0, energy * 10))
        label = "AI VOICE" if confidence > 0.5 else "HUMAN VOICE"

        return {
            "type": "audio",
            "label": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "type": "audio",
            "label": "ERROR",
            "confidence": 0.0,
            "error": str(e)
        }
