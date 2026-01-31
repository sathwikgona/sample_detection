import io
from PIL import Image
import numpy as np
import cv2
from fastapi import UploadFile

# Load face detector once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

async def predict_image(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Face detection (relaxed)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    face_count = len(faces)

    # Brightness (fallback)
    brightness = gray.mean() / 255.0

    # ---- FINAL DECISION ----
    if face_count > 0:
        label = "REAL"
        confidence = round(0.85 + min(0.1, face_count * 0.05), 2)
    else:
        if brightness < 0.25:
            label = "FAKE"
            confidence = 0.90
        else:
            label = "REAL"
            confidence = 0.70

    return {
        "type": "image",
        "label": label,
        "confidence": confidence,
        "faces_detected": face_count,
        "brightness": round(brightness, 2)
    }
