from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import shutil

app = FastAPI()
model = WhisperModel("base")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, info = model.transcribe(file_path)

    text = ""
    for segment in segments:
        text += segment.text + "\n"

    return {
        "language": info.language,
        "text": text
    }