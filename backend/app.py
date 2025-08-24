# # backend/app.py

import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment variable
port = int(os.environ.get("PORT", 8080))
logger.info(f"Server starting on port: {port}")

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Emotion Detection API", version="1.0.0")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------------------
# Face detection - Download Haar cascade if missing
# -------------------------------
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")

# Download Haar cascade if not exists
if not os.path.exists(cascade_path):
    logger.warning("Haar cascade file not found, downloading...")
    import urllib.request
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(cascade_url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

# -------------------------------
# Health check endpoints
# -------------------------------
@app.get("/")
async def root():
    return {
        "message": "Emotion Detection API",
        "status": "running",
        "model_loaded": False,
        "note": "Model download excluded for testing"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_ready": False}

@app.get("/test")
async def test_endpoint():
    return {
        "message": "Backend is working!",
        "endpoints": {
            "health": "/health",
            "root": "/",
            "predict": "/predict (POST)"
        }
    }

# -------------------------------
# Mock Predict endpoint (without model)
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return {
                "emotion": "No face detected", 
                "confidence": 0.0, 
                "all_predictions": {},
                "note": "Model not loaded - this is a mock response"
            }

        # If face is detected, return mock prediction
        x, y, w, h = faces[0]
        
        # Mock prediction response
        result = {
            "emotion": "Happy",  # Mock emotion
            "confidence": 85.5,  # Mock confidence
            "all_predictions": {
                "Angry": 0.1,
                "Disgust": 0.05,
                "Fear": 0.05,
                "Happy": 0.85,
                "Sad": 0.02,
                "Surprise": 0.01,
                "Neutral": 0.02
            },
            "note": "This is a mock response - model not loaded"
        }

        return result

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# -------------------------------
# Additional test endpoint for face detection only
# -------------------------------
@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """Endpoint to test only face detection without prediction"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return {
            "faces_detected": len(faces),
            "face_coordinates": [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces],
            "message": "Face detection working correctly"
        }

    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error in face detection: {str(e)}")
    
















    
# import os
# import zipfile
# import gdown
# import tensorflow as tf
# import io
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image, ImageFilter
# import cv2


# # Get port from environment variable (optional, for info)
# port = int(os.environ.get("PORT", 8080))
# print(f"Server starting on port: {port}")
# # -------------------------------
# # Google Drive model download
# # -------------------------------

# # Google Drive file ID for the zip of model-hopeful-meadow-1v12
# # Replace this with your own file ID
# GOOGLE_DRIVE_FILE_ID = "1I6jM9cFJ9zfwvOYHZNChcvQqjhQ4vO-M"
# LOCAL_MODEL_DIR = "artifacts/model-hopeful-meadow-1v12"
# ZIP_PATH = "artifacts/model.zip"

# def download_model_from_drive():
#     if os.path.exists(LOCAL_MODEL_DIR):
#         print("Model already exists locally.")
#         return

#     os.makedirs("artifacts", exist_ok=True)

#     # Construct download URL
#     # https://drive.google.com/file/d/1I6jM9cFJ9zfwvOYHZNChcvQqjhQ4vO-M/view
#     url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
#     print("Downloading model from Google Drive...")
#     gdown.download(url, ZIP_PATH, quiet=False)

#     # Unzip
#     print("Extracting model...")
#     with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
#         zip_ref.extractall("artifacts")

#     os.remove(ZIP_PATH)
#     print("Model ready at", LOCAL_MODEL_DIR)

# # Call before loading model
# download_model_from_drive()

# # -------------------------------
# # Load TensorFlow model
# # -------------------------------
# model = tf.saved_model.load(LOCAL_MODEL_DIR)
# infer = model.signatures["serving_default"]

# # -------------------------------
# # FastAPI setup
# # -------------------------------
# app = FastAPI()

# # Allow frontend requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # adjust for frontend URL later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # -------------------------------
# # Face detection
# # -------------------------------
# cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier(cascade_path)

# if face_cascade.empty():
#     raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

# # -------------------------------
# # Predict endpoint
# # -------------------------------
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         if not file.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="File must be an image")

#         image_bytes = await file.read()
#         np_arr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if img is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")

#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         if len(faces) == 0:
#             return {"emotion": "No face detected", "confidence": 0.0, "all_predictions": {}}

#         x, y, w, h = faces[0]
#         face_img = gray_img[y:y+h, x:x+w]
#         face_img_resized = cv2.resize(face_img, (48, 48))

#         pil_img = Image.fromarray(face_img_resized)
#         sharpened_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
#         sharpened_img = np.array(sharpened_img)

#         sharpened_img = sharpened_img.astype("float32") / 255.0
#         sharpened_img = np.expand_dims(sharpened_img, axis=(0, -1))

#         tensor_input = tf.convert_to_tensor(sharpened_img, dtype=tf.float32)
#         outputs = infer(tf.constant(tensor_input))

#         predictions = list(outputs.values())[0].numpy()
#         predicted_class = int(np.argmax(predictions, axis=1)[0])
#         predicted_prob = float(predictions[0][predicted_class])

#         result = {
#             "emotion": emotion_labels[predicted_class],
#             "confidence": round(predicted_prob * 100, 2),
#             "all_predictions": {emotion_labels[i]: float(predictions[0][i]) for i in range(len(emotion_labels))}
#         }

#         return result

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
