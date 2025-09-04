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
import math
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

# Head movement tracking
previous_head_pose = None
head_movement_history = deque(maxlen=10)

def calculate_head_pose(landmarks, image_shape):
    """Calculate head pose angles from facial landmarks"""
    image_height, image_width = image_shape[:2]
    
    # Key landmarks for head pose estimation
    nose_tip = landmarks.landmark[1]
    chin = landmarks.landmark[152]
    left_eye_left_corner = landmarks.landmark[33]
    right_eye_right_corner = landmarks.landmark[263]
    left_mouth_corner = landmarks.landmark[61]
    right_mouth_corner = landmarks.landmark[291]
    
    # Convert to pixel coordinates
    nose = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
    chin_point = (int(chin.x * image_width), int(chin.y * image_height))
    left_eye = (int(left_eye_left_corner.x * image_width), int(left_eye_left_corner.y * image_height))
    right_eye = (int(right_eye_right_corner.x * image_width), int(right_eye_right_corner.y * image_height))
    left_mouth = (int(left_mouth_corner.x * image_width), int(left_mouth_corner.y * image_height))
    right_mouth = (int(right_mouth_corner.x * image_width), int(right_mouth_corner.y * image_height))
    
    # Calculate head pose angles (simplified)
    # Yaw: horizontal head rotation
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    face_center_x = image_width / 2
    yaw_angle = (eye_center_x - face_center_x) / (image_width / 2) * 45  # ±45 degrees
    
    # Pitch: vertical head rotation
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    face_vertical_center = (eye_center_y + mouth_center_y) / 2
    image_center_y = image_height / 2
    pitch_angle = (face_vertical_center - image_center_y) / (image_height / 2) * 45  # ±45 degrees
    
    return {
        'yaw': yaw_angle,
        'pitch': pitch_angle,
        'nose_position': nose
    }

def estimate_head_movement(current_pose):
    """Estimate head movement percentage based on pose changes"""
    global previous_head_pose, head_movement_history
    
    if previous_head_pose is None:
        previous_head_pose = current_pose
        return 0
    
    # Calculate movement between frames
    yaw_diff = abs(current_pose['yaw'] - previous_head_pose['yaw'])
    pitch_diff = abs(current_pose['pitch'] - previous_head_pose['pitch'])
    
    # Calculate Euclidean distance of nose movement
    nose_movement = math.sqrt(
        (current_pose['nose_position'][0] - previous_head_pose['nose_position'][0])**2 +
        (current_pose['nose_position'][1] - previous_head_pose['nose_position'][1])**2
    )
    
    # Normalize movement (0-100 scale)
    movement_score = min(100, (yaw_diff + pitch_diff + nose_movement / 10) * 5)
    
    previous_head_pose = current_pose
    head_movement_history.append(movement_score)
    
    # Return average of recent movements
    return sum(head_movement_history) / len(head_movement_history) if head_movement_history else 0

def estimate_eye_contact_mediapipe(face_img_color):
    """Returns eye contact 0-100 using MediaPipe FaceMesh."""
    eye_contact = 0
    rgb_face = cv2.cvtColor(face_img_color, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh_instance.process(rgb_face)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_iris_x = np.mean([l.x for l in landmarks.landmark[468:473]])
        right_iris_x = np.mean([l.x for l in landmarks.landmark[473:478]])
        
        # Calculate head pose for movement detection
        head_pose = calculate_head_pose(landmarks, face_img_color.shape)
        head_movement = estimate_head_movement(head_pose)
        
        # Relax threshold
        if 0.3 < left_iris_x < 0.7 and 0.3 < right_iris_x < 0.7:
            eye_contact = 100
        else:
            eye_contact = 50  # partially looking at camera
    else:
        eye_contact = 0  # no face/iris detected
        head_movement = 0

    return eye_contact, head_movement

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
                "eye_contact": 0,
                "head_movement": 0
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
            eye_contact_percentage, head_movement_percentage = estimate_eye_contact_mediapipe(face_img_color)
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "all_predictions": {},
                "eye_contact": eye_contact_percentage,
                "head_movement": head_movement_percentage
            }

        predictions = predictions_arr  # shape: (1, num_classes)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        # Guard against unexpected class size
        if predictions.shape[1] != len(emotion_labels):
            eye_contact_percentage, head_movement_percentage = estimate_eye_contact_mediapipe(face_img_color)
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "all_predictions": {},
                "eye_contact": eye_contact_percentage,
                "head_movement": head_movement_percentage
            }

        predicted_prob = float(predictions[0][predicted_class])

        # Eye contact and head movement using MediaPipe
        eye_contact_percentage, head_movement_percentage = estimate_eye_contact_mediapipe(face_img_color)

        return {
            "emotion": emotion_labels[predicted_class],
            "confidence": round(predicted_prob*100,2),
            "all_predictions": {emotion_labels[i]: float(predictions[0][i]) for i in range(len(emotion_labels))},
            "eye_contact": eye_contact_percentage,
            "head_movement": head_movement_percentage
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Emotion Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)