from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, vad_filter=True)
        text = "".join([seg.text for seg in segments]).strip()
        return {"text": text, "language": info.language}
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
