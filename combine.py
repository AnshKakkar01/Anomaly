from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import requests
import io
import os
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend URL or other origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Custom Vision details
PREDICTION_KEY = os.getenv("PREDICTION_KEY")
ENDPOINT = os.getenv("ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
ITERATION_NAME = os.getenv("ITERATION_NAME")
PREDICTION_URL = os.getenv("PREDICTION_URL", f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/detect/iterations/{ITERATION_NAME}/image")

class ImageURL(BaseModel):
    url: str

def get_prediction_from_stream(image_stream):
    headers = {
        'Prediction-Key': PREDICTION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    response = requests.post(PREDICTION_URL, headers=headers, data=image_stream)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

def get_prediction_from_url(image_url):
    headers = {
        'Prediction-Key': PREDICTION_KEY,
        'Content-Type': 'application/json'
    }
    body = {'Url': image_url}
    response = requests.post(PREDICTION_URL, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# def apply_anomaly_logic(predictions):
#     security_weapons_detected = False
#     civilian_weapons_detected = False
#     security_detected = False
#     civilian_detected = False

    if not predictions:
        return {"anomaly": True, "reason": "No significant objects detected, anomaly."}

    for prediction in predictions:
        tag_name = prediction['tagName'].lower()

        if tag_name == 'security_weapons':
            security_weapons_detected = True
        if tag_name == 'civilian_weapons':
            civilian_weapons_detected = True
        if tag_name == 'security':
            security_detected = True
        if tag_name == 'civilian':
            civilian_detected = True

    if security_weapons_detected or security_detected:
        return {"anomaly": False, "reason": "Security presence detected, no anomaly."}
    elif civilian_weapons_detected and civilian_detected:
        return {"anomaly": True, "reason": "Civilian with weapon detected, anomaly."}
    elif civilian_weapons_detected:
        return {"anomaly": True, "reason": "Civilian weapon detected, anomaly."}
    elif civilian_detected:
        return {"anomaly": False, "reason": "Civilian detected, no anomaly."}

    return {"anomaly": True, "reason": "Unclassified situation, anomaly."}

@app.post("/detect-humans/")
async def detect_humans(image: UploadFile = File(...)):
    image_content = await image.read()
    image_stream = io.BytesIO(image_content)

    # Get prediction
    prediction = get_prediction_from_stream(image_stream)

    # Rewind the stream and load the image for drawing
    image_stream.seek(0)
    img = Image.open(image_stream).convert("RGBA")
    draw = ImageDraw.Draw(img)

    human_count = 0

    for pred in prediction['predictions']:
        if pred['tagName'].lower() == 'person' and pred['probability'] > 0.80:
            human_count += 1
            bbox = pred['boundingBox']
            left = bbox['left'] * img.width
            top = bbox['top'] * img.height
            right = left + (bbox['width'] * img.width)
            bottom = top + (bbox['height'] * img.height)

            draw.rectangle([left, top, right, bottom], outline="red", width=2)

    # Save the image with highlighted detections
    highlighted_image_path = "highlighted_image.png"
    img.save(highlighted_image_path)

    # Categorize the crowd
    if human_count <= 10:
        crowd_category = "not crowded"
    else:
        crowd_category = "crowded"

    return {
        "human_count": human_count,
        "crowd_category": crowd_category,
        "highlighted_image": highlighted_image_path  # You might want to serve this image through a proper URL or storage
    }

@app.post("/detect-humans-url/")
async def detect_humans_url(image_url: ImageURL):
    prediction = get_prediction_from_url(image_url.url)

    # Count the number of humans detected
    human_count = sum(1 for pred in prediction['predictions'] if pred['tagName'].lower() == 'person' and pred['probability'] > 0.5)

    # Categorize the crowd
    if human_count < 10:
        crowd_category = "not crowded"
    elif 10 <= human_count <= 18:
        crowd_category = "moderately crowded"
    else:
        crowd_category = "heavily crowded"

    return {
        "human_count": human_count,
        "crowd_category": crowd_category
    }

@app.post("/predict/")
async def predict_image(image: UploadFile = File(...)):
    image_content = await image.read()
    image_stream = io.BytesIO(image_content)
    prediction = get_prediction_from_stream(image_stream)

    # Apply threshold
    filtered_predictions = [p for p in prediction["predictions"] if p["probability"] > 0.65]

    # Apply anomaly logic
    result = apply_anomaly_logic(filtered_predictions)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
