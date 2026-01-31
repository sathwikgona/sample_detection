async def predict_text(text: str):
    wc = len(text.split())

    if wc > 30:
        return {
            "type": "text",
            "label": "AI GENERATED",
            "confidence": 0.95
        }
    else:
        return {
            "type": "text",
            "label": "HUMAN WRITTEN",
            "confidence": 0.95
        }
