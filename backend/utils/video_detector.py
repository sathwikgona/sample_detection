import cv2
import numpy as np
import tempfile
from fastapi import UploadFile
import torch
from torchvision import models, transforms

device = torch.device("cpu")

# Load pretrained CNN for frame analysis
model = models.resnet18(pretrained=True)
model.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

async def predict_video(file: UploadFile):
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(await file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1).max().item()
            frame_preds.append(prob)

    cap.release()

    final_score = float(np.mean(frame_preds)) if frame_preds else 0.0
    label = "FAKE" if final_score > 0.5 else "REAL"

    return {
        "type": "video",
        "label": label,
        "confidence": round(final_score, 4),
        "frames_analyzed": len(frame_preds)
    }
