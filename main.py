from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

PORT = 8000

app = FastAPI()

with open("disease.json") as result:
    DISEASES_DETAILS = json.load(result)

MODEL = tf.keras.models.load_model("rice.keras")

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
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100

    details = DISEASES_DETAILS.get(predicted_class, {
        "Disease": "No details available",
		"Description": "Unknown",
		"Recommendations": "N/A",
		"Pesticide": "N/A",
		"Guidelines": "N/A"
    })

    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}",
        "details": details
    }


