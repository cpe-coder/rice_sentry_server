from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:3000/v1/models/rice_model:predict"

MODEL = tf.keras.models.load_model("../models/rice.keras")

CLASS_NAMES = [
    'ArmyWorm', 'BackterialBlight', 'BrownPlanthopper', 'BrownSpot', 'CaseWorm',
    'FalseSmut', 'GoldenAppleSnail', 'GreenLeafHopper', 'LeafScald', 'RiceBlast',
    'RiceDiscoloration', 'RiceEarBug', 'RiceGallMidge', 'RiceLeafFolder', 'RiceMealyBug',
    'RiceRootAphid', 'RiceStemBorer', 'RiceThrips', 'RiceWhorlMaggot', 'ShealthBlight',
    'SheathRot', 'WhitePlanthopper', 'ZigzagLeafhopper', 'healty'
]

IMAGE_SIZE = 256  

@app.get("/")
async def root():
    return {"message": "Hello World"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)  
    
    json_data ={
        "instance": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    predictions = np.array(response.json()["predictions"][0])


    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100

    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
