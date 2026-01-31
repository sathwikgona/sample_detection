import base64
import tempfile
from fastapi import WebSocket
from .audio_detector import predict_audio

async def handle_mic_stream(websocket: WebSocket):
    await websocket.accept()
    print("Mic client connected")

    try:
        while True:
            data = await websocket.receive_text()
            audio_bytes = base64.b64decode(data)

            # Create a temporary audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                temp_audio_path = f.name

            # Create dummy UploadFile-like object
            class DummyAudio:
                def __init__(self, path):
                    self.path = path
                async def read(self):
                    with open(self.path, "rb") as af:
                        return af.read()

            dummy_file = DummyAudio(temp_audio_path)
            result = await predict_audio(dummy_file)

            await websocket.send_json(result)

    except Exception as e:
        print("Mic stream closed:", e)
