import streamlit as st
import cv2
import io
import numpy as np
import requests
from collections import deque, defaultdict
from PIL import Image

# =========================
# CONFIG
# =========================
BACKEND_URL = "http://localhost:8000/predict"  # Adjust if backend port differs
ROLLING_WINDOW = 15
FPS = 15  # For video capture

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
if "eye_contact_history" not in st.session_state:
    st.session_state.eye_contact_history = deque(maxlen=ROLLING_WINDOW)
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
    r = requests.post(BACKEND_URL, files=files, timeout=30)
    r.raise_for_status()
    return r.json()

def update_rollups(emotion_result):
    # Update emotion history
    probs = emotion_result.get("all_predictions", {})
    full = {e: 0.0 for e in EMOTIONS}
    for k, v in probs.items():
        mapped = map_emotion(k)
        full[mapped] += float(v)
    st.session_state.history.append(full)

    # Update eye contact history
    eye_contact_level = emotion_result.get("eye_contact", 0)
    st.session_state.eye_contact_history.append(eye_contact_level)

    # Compute averages
    emotion_avg = defaultdict(float)
    for row in st.session_state.history:
        for k, v in row.items():
            emotion_avg[k] += v
    n = max(1, len(st.session_state.history))
    for k in emotion_avg:
        emotion_avg[k] /= n

    eye_contact_avg = sum(st.session_state.eye_contact_history) / max(1, len(st.session_state.eye_contact_history))

    # Determine overall eye contact performance
    if eye_contact_avg >= 80:
        performance_label = "Good"
    elif eye_contact_avg >= 40:
        performance_label = "Average"
    else:
        performance_label = "Bad"

    return emotion_avg, eye_contact_avg, performance_label

# =========================
# UI
# =========================
st.set_page_config(page_title="Real-time Emotion & Eye Contact", page_icon="😊", layout="wide")
st.title("😊 Real-time Emotion & Eye Contact Detection")
st.write("Webcam captures emotions and eye contact using Haar + MediaPipe.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera", disabled=st.session_state.camera_active):
        st.session_state.camera_active = True
        st.session_state.history.clear()
        st.session_state.eye_contact_history.clear()
        st.rerun()
with col2:
    if st.button("Stop Camera", disabled=not st.session_state.camera_active):
        st.session_state.camera_active = False
        st.rerun()

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
        st.session_state.camera_active = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        video_placeholder = st.empty()
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        details_placeholder = st.empty()

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Display webcam feed
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width=500)

            try:
                # Send frame to backend
                result = post_to_backend(frame)
                emotion_avg, eye_contact_avg, eye_contact_perf = update_rollups(result)

                # Display current emotion and eye contact
                current_emotion = map_emotion(result.get("emotion","None"))
                confidence = result.get("confidence", 0)
                eye_contact = result.get("eye_contact", 0)
                status_placeholder.success(f"Emotion: **{current_emotion}** ({confidence}%) | Eye Contact: **{eye_contact}**")

                # Display rolling averages
                sorted_emotion_avg = sorted(emotion_avg.items(), key=lambda kv: kv[1], reverse=True)
                details_text = "**Average Emotion Percentages:**\n\n"
                for e, v in sorted_emotion_avg:
                    details_text += f"- {e}: {round(v*100,1)}%\n"
                details_text += f"\n**Average Eye Contact Level:** {round(eye_contact_avg,1)}"
                details_text += f"\n**Eye Contact Performance:** {eye_contact_perf}"
                details_placeholder.markdown(details_text)

                # Bar chart
                chart_data = {k:[v] for k,v in emotion_avg.items()}
                chart_placeholder.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Backend error: {e}")

else:
    st.info("Click 'Start Camera' to begin detection")

# Reset button
if st.button("Reset Averages"):
    st.session_state.history.clear()
    st.session_state.eye_contact_history.clear()
    st.success("Rolling averages cleared")
