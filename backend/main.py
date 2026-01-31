from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from utils.image_detector import predict_image
from utils.video_detector import predict_video
from utils.audio_detector import predict_audio
from utils.text_detector import predict_text
from utils.stream_camera import handle_camera_stream
from utils.stream_mic import handle_mic_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Server running", "message": "Multi-Modal AI Backend Ready"}

@app.post("/predict/image")
async def image_api(file: UploadFile = File(...)):
    return await predict_image(file)

@app.post("/predict/video")
async def video_api(file: UploadFile = File(...)):
    return await predict_video(file)

@app.post("/predict/audio")
async def audio_api(file: UploadFile = File(...)):
    return await predict_audio(file)

@app.post("/predict/text")
async def text_api(payload: dict):
    return await predict_text(payload["text"])

@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket):
    await handle_camera_stream(websocket)

@app.websocket("/ws/mic")
async def mic_ws(websocket: WebSocket):
    await handle_mic_stream(websocket)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
