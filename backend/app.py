# backend/app.py
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
import cv2

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load SavedModel
artifact_dir = "artifacts/model-hopeful-meadow-1v12"
model = tf.saved_model.load(artifact_dir)

# Get inference function (depends on how model was exported)
infer = model.signatures["serving_default"]
import os

cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate input
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read uploaded file into OpenCV format
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return {"emotion": "No face detected", "confidence": 0.0, "all_predictions": {}}

        # Take first face
        x, y, w, h = faces[0]
        face_img = gray_img[y:y+h, x:x+w]

        # Resize → 48x48
        face_img_resized = cv2.resize(face_img, (48, 48))

        # Sharpen
        pil_img = Image.fromarray(face_img_resized)
        sharpened_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        sharpened_img = np.array(sharpened_img)

        # Normalize
        sharpened_img = sharpened_img.astype("float32") / 255.0
        sharpened_img = np.expand_dims(sharpened_img, axis=(0, -1))  # shape: (1,48,48,1)

        # Run inference
        tensor_input = tf.convert_to_tensor(sharpened_img, dtype=tf.float32)
        outputs = infer(tf.constant(tensor_input))
        
        # Extract predictions (depending on output key)
        predictions = list(outputs.values())[0].numpy()  # shape (1, 7)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        predicted_prob = float(predictions[0][predicted_class])

        result = {
            "emotion": emotion_labels[predicted_class],
            "confidence": round(predicted_prob * 100, 2),
            "all_predictions": {emotion_labels[i]: float(predictions[0][i]) for i in range(len(emotion_labels))}
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
