from fastapi import WebSocket
import json

async def handle_camera_stream(websocket: WebSocket):
    await websocket.accept()
    print("Camera WebSocket connected")

    while True:
        data = await websocket.receive_text()  # receive frame
        # Just send a dummy response
        response = {
            "label": "CONNECTED",
            "confidence": 1.0
        }
        await websocket.send_text(json.dumps(response))
