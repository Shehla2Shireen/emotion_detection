import os
import zipfile
import gdown
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
import cv2
import mediapipe as mp
from collections import deque

# -------------------------------
# Model download & load
# -------------------------------
GOOGLE_DRIVE_FILE_ID = "1I6jM9cFJ9zfwvOYHZNChcvQqjhQ4vO-M"
LOCAL_MODEL_DIR = "artifacts/model-hopeful-meadow-1v12"
ZIP_PATH = "artifacts/model.zip"

def download_model_from_drive():
    if os.path.exists(LOCAL_MODEL_DIR):
        return
    os.makedirs("artifacts", exist_ok=True)
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("artifacts")
    os.remove(ZIP_PATH)

download_model_from_drive()
model = tf.saved_model.load(LOCAL_MODEL_DIR)
infer = model.signatures["serving_default"]

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_instance = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def estimate_eye_contact_mediapipe(face_img_color):
    """Returns eye contact 0-100 using MediaPipe FaceMesh."""
    eye_contact = 0
    rgb_face = cv2.cvtColor(face_img_color, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh_instance.process(rgb_face)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_iris_x = np.mean([l.x for l in landmarks.landmark[468:473]])
        right_iris_x = np.mean([l.x for l in landmarks.landmark[473:478]])
        
        # Relax threshold
        if 0.27 < left_iris_x < 0.73 and 0.27 < right_iris_x < 0.73:
            eye_contact = 100
        else:
            eye_contact = 50  # partially looking at camera
    else:
        eye_contact = 0  # no face/iris detected

    return eye_contact

# -------------------------------
# Predict endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if len(faces) == 0:
            return {
                "emotion": "No face detected", 
                "confidence": 0.0, 
                "all_predictions": {}, 
                "eye_contact": 0
            }

        x, y, w, h = faces[0]
        face_img_gray = gray_img[y:y+h, x:x+w]
        face_img_color = img[y:y+h, x:x+w]

        # Emotion preprocessing
        face_resized = cv2.resize(face_img_gray, (48,48))
        pil_img = Image.fromarray(face_resized)
        sharpened_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        sharpened_img = np.array(sharpened_img).astype("float32") / 255.0
        sharpened_img = np.expand_dims(sharpened_img, axis=(0,-1))
        tensor_input = tf.convert_to_tensor(sharpened_img, dtype=tf.float32)
        outputs = infer(tf.constant(tensor_input))

        # FIX: defensive check on model outputs to avoid index errors
        predictions_arr = list(outputs.values())[0].numpy()
        if predictions_arr.ndim != 2 or predictions_arr.shape[0] == 0:
            eye_contact_percentage = estimate_eye_contact_mediapipe(face_img_color)
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "all_predictions": {},
                "eye_contact": eye_contact_percentage
            }

        predictions = predictions_arr  # shape: (1, num_classes)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        # Guard against unexpected class size
        if predictions.shape[1] != len(emotion_labels):
            eye_contact_percentage = estimate_eye_contact_mediapipe(face_img_color)
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "all_predictions": {},
                "eye_contact": eye_contact_percentage
            }

        predicted_prob = float(predictions[0][predicted_class])

        # Eye contact using MediaPipe
        eye_contact_percentage = estimate_eye_contact_mediapipe(face_img_color)

        return {
            "emotion": emotion_labels[predicted_class],
            "confidence": round(predicted_prob*100,2),
            "all_predictions": {emotion_labels[i]: float(predictions[0][i]) for i in range(len(emotion_labels))},
            "eye_contact": eye_contact_percentage
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Emotion Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
