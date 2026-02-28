import os
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI()

# Modèle + config "CPU friendly" (stable en gratuit)
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # tiny / base / small
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(None), url: str = None):
    """
    Envoie soit:
      - un fichier audio (multipart) -> file
      - une URL téléchargeable -> url
    Retour: {"text": "..."}
    """
    if file is None and not url:
        return JSONResponse({"error": "Provide 'file' or 'url'."}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp_path = tmp.name

    try:
        if url:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(r.content)
        else:
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

        segments, info = model.transcribe(tmp_path)
        text = "".join([s.text for s in segments]).strip()
        return {"text": text, "language": info.language}
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
