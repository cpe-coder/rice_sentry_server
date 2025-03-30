import base64
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True) 

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI file upload service!"}


class ImageUpload(BaseModel):
    filename: str
    image_base64: str

@app.post("/upload/")
async def upload_file(data: ImageUpload):
    file_path = os.path.join(UPLOAD_DIR, data.filename)

    with open(file_path, "wb") as file:
        file.write(base64.b64decode(data.image_base64))

    return JSONResponse(content={"filename": data.filename, "message": "File uploaded successfully"})