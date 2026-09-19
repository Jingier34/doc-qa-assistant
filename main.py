from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")
    return {
        "filename": file.filename,
        "text_length": len(text),
        "preview": text[:200]
    }