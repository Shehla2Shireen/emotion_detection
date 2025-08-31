import streamlit as st
import cv2
import io
import time
import tempfile
import numpy as np
import requests
from collections import deque, defaultdict
from PIL import Image

# =========================
# CONFIG
# =========================
API_URL = "http://localhost:8000/predict"
SEND_INTERVAL_SEC = 2.0
ROLLING_WINDOW = 15

# Original emotions from backend
ORIGINAL_EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
# Map Disgust and Fear to Sad
EMOTIONS = ['Angry','Happy','Sad','Surprise','Neutral']
MAP_TO_SAD = {'Fear': 'Sad', 'Disgust': 'Sad'}

# =========================
# STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=ROLLING_WINDOW)
if "last_send_ts" not in st.session_state:
    st.session_state.last_send_ts = 0.0
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# =========================
# HELPERS
# =========================
def map_emotion(emotion: str) -> str:
    return MAP_TO_SAD.get(emotion, emotion)

def post_to_backend(img_bgr: np.ndarray):
    _, buf = cv2.imencode(".jpg", img_bgr)
    files = {"file": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    r = requests.post(API_URL, files=files, timeout=30, verify=False)
    r.raise_for_status()
    return r.json()

def update_rollups(result):
    probs = result.get("all_predictions", {})
    # Remap Disgust/Fear -> Sad
    full = {e: 0.0 for e in EMOTIONS}
    for k, v in probs.items():
        mapped = map_emotion(k)
        full[mapped] += float(v)
    st.session_state.history.append(full)

    avg = defaultdict(float)
    for row in st.session_state.history:
        for k, v in row.items():
            avg[k] += v
    n = max(1, len(st.session_state.history))
    for k in avg:
        avg[k] /= n
    return avg

# =========================
# UI
# =========================
st.set_page_config(page_title="Real-time Emotion Detection", page_icon="😊", layout="wide")
st.title("😊 Real-time Emotion Detection App")
st.write("This app captures images from your webcam every 2 seconds and analyzes emotions.")
st.markdown("---")

# -------------------------
# WEBCAM CAPTURE
# -------------------------
st.subheader("Webcam Emotion Detection")

# Start/Stop camera buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera", disabled=st.session_state.camera_active):
        st.session_state.camera_active = True
        st.session_state.history.clear()  # Clear history when starting
        st.rerun()

with col2:
    if st.button("Stop Camera", disabled=not st.session_state.camera_active):
        st.session_state.camera_active = False
        st.rerun()

if st.session_state.camera_active:
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check permissions.")
        st.session_state.camera_active = False
    else:
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        video_placeholder = col1.empty()
        status_placeholder = col2.empty()
        chart_placeholder = col2.empty()
        details_placeholder = col2.empty()
        
        # Initial state
        last_emotion = "None"
        last_confidence = 0
        
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
                
            # Display the webcam feed
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width=500)
            
            # Process frame every SEND_INTERVAL_SEC seconds
            now = time.time()
            if now - st.session_state.last_send_ts >= SEND_INTERVAL_SEC:
                try:
                    # Send to backend
                    result = post_to_backend(frame)
                    result['emotion'] = map_emotion(result['emotion'])
                    
                    # Update history and get averages
                    avg = update_rollups(result)
                    st.session_state.last_send_ts = now
                    
                    # Update last emotion and confidence
                    last_emotion = result['emotion']
                    last_confidence = result['confidence']
                    
                    # Display current emotion
                    status_placeholder.success(f"Current: **{last_emotion}** ({last_confidence}%)")
                    
                    # Display rolling average
                    sorted_avg = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
                    dominant_emotion = sorted_avg[0][0]
                    dominant_percent = round(sorted_avg[0][1] * 100, 1)
                    
                    # Display emotion percentages from most to least dominant
                    details_text = "**Emotion Percentages:**\n\n"
                    for emotion, percentage in sorted_avg:
                        details_text += f"- {emotion}: {round(percentage * 100, 1)}%\n"
                    
                    details_placeholder.markdown(details_text)
                    
                    # Display bar chart
                    chart_data = {k: [v] for k, v in avg.items()}
                    chart_placeholder.bar_chart(chart_data)
                    
                except Exception as e:
                    status_placeholder.error(f"Request failed: {e}")
            
            # Add small delay to prevent high CPU usage
            time.sleep(0.05)
        
        # Release camera when done
        cap.release()
else:
    st.info("Click 'Start Camera' to begin real-time emotion detection")

# =========================
# RESET BUTTON
# =========================
if st.button("Reset rolling average"):
    st.session_state.history.clear()
    st.success("Averages cleared.")