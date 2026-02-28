from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# Choix pro/stable pour CPU : "small" ou "base"
# - base: + léger, + rapide, bonne qualité
# - small: meilleure qualité, plus lent
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

# CPU only (Railway free tier = CPU). compute_type int8 = plus léger.
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_SIZE}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "fr"):
    # Sauvegarde temporaire
    suffix = os.path.splitext(file.filename)[1] or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            vad_filter=True,      # filtre silence/bruit
            beam_size=5
        )

        text = " ".join([seg.text.strip() for seg in segments]).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": info.duration
        }
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
