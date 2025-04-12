from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load your model
MODEL = tf.keras.models.load_model("../models/rice.h5")

# Class names
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
    confidence = np.max(predictions[0])
    print(predicted_class)
    print(confidence)
    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
