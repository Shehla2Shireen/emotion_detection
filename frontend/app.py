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
BACKEND_URL = "http://localhost:8000/predict"
ROLLING_WINDOW = 15
FPS = 15

# Original emotions from backend
ORIGINAL_EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
EMOTIONS = ['Angry','Happy','Sad','Surprise','Neutral','Disgust','Fear']
# MAP_TO_SAD = {'Fear': 'Sad', 'Disgust': 'Sad'}  # Commented out but kept for reference

# Stress calculation parameters (research-based weights)
STRESS_WEIGHTS = {
    'negative_emotions': 0.4,      # Fear, Anger, Sad
    'eye_contact': 0.3,            # Reduced eye contact
    'head_movement': 0.3           # Increased head movement
}

# =========================
# STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=ROLLING_WINDOW)
if "eye_contact_history" not in st.session_state:
    st.session_state.eye_contact_history = deque(maxlen=ROLLING_WINDOW)
if "head_movement_history" not in st.session_state:
    st.session_state.head_movement_history = deque(maxlen=ROLLING_WINDOW)
if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque(maxlen=ROLLING_WINDOW)
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# =========================
# HELPERS
# =========================
# def map_emotion(emotion: str) -> str:
#     """Map Fear and Disgust to Sad for simplified emotion analysis"""
#     return MAP_TO_SAD.get(emotion, emotion)  # Commented out but kept for reference

# =========================
# STRESS LABEL HELPER
# =========================
def get_stress_label(stress_score: float):
    """
    Convert numerical stress score into a label + emoji.
    """
    if stress_score < 40:
        return "Low Stress", "🟢"
    elif stress_score < 65:
        return "Moderate Stress", "🟡"
    else:
        return "High Stress", "🔴"

def post_to_backend(img_bgr: np.ndarray):
    _, buf = cv2.imencode(".jpg", img_bgr)
    files = {"file": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    r = requests.post(BACKEND_URL, files=files, timeout=30)
    r.raise_for_status()
    return r.json()

# =========================
# IMPROVED STRESS CALCULATION
# =========================

# Emotion-specific weights (based on stress literature: fear > anger > sadness)
EMOTION_WEIGHTS = {
    'Fear': 1.0,
    'Angry': 0.8,
    'Sad': 0.6
}

# Overall factor weights (how much each group contributes to stress)
STRESS_WEIGHTS = {
    'emotions': 0.45,       # emotional cues are strongest
    'eye_contact': 0.30,    # reduced eye contact
    'head_movement': 0.25   # jittery movements
}

# Smoothing factor for exponential moving average (EMA)
ALPHA = 0.3

if "smoothed_stress" not in st.session_state:
    st.session_state.smoothed_stress = 0

def calculate_stress_level(emotions, eye_contact, head_movement):
    """
    Calculate stress level based on weighted behavioral indicators
    """
    # Weighted negative emotions
    neg_score = 0
    for emo, weight in EMOTION_WEIGHTS.items():
        neg_score += emotions.get(emo, 0) * weight
    
    # Normalize emotion component (0–100)
    emotion_component = min(100, neg_score * 100)

    # Eye contact (inverse relationship, 0 = bad, 100 = good)
    eye_contact_component = min(100, (100 - eye_contact))

    # Head movement (more = stressed/jittery)
    head_movement_component = min(100, head_movement)

    # Weighted sum of components
    raw_stress = (
        STRESS_WEIGHTS['emotions'] * emotion_component +
        STRESS_WEIGHTS['eye_contact'] * eye_contact_component +
        STRESS_WEIGHTS['head_movement'] * head_movement_component
    )

    # Exponential Moving Average smoothing
    smoothed = ALPHA * raw_stress + (1 - ALPHA) * st.session_state.smoothed_stress
    st.session_state.smoothed_stress = smoothed

    return smoothed

def update_rollups(emotion_result):
    # Update emotion history
    probs = emotion_result.get("all_predictions", {})
    full = {e: 0.0 for e in EMOTIONS}
    for k, v in probs.items():
        # mapped = map_emotion(k)  # Commented out mapping functionality
        mapped = k  # Use original emotion without mapping
        full[mapped] += float(v)
    st.session_state.history.append(full)

    # Update eye contact history
    eye_contact_level = emotion_result.get("eye_contact", 0)
    st.session_state.eye_contact_history.append(eye_contact_level)

    # Update head movement history
    head_movement_level = emotion_result.get("head_movement", 0)
    st.session_state.head_movement_history.append(head_movement_level)

    # Compute averages
    emotion_avg = defaultdict(float)
    for row in st.session_state.history:
        for k, v in row.items():
            emotion_avg[k] += v
    n = max(1, len(st.session_state.history))
    for k in emotion_avg:
        emotion_avg[k] /= n

    eye_contact_avg = sum(st.session_state.eye_contact_history) / max(1, len(st.session_state.eye_contact_history))
    head_movement_avg = sum(st.session_state.head_movement_history) / max(1, len(st.session_state.head_movement_history))

    # Calculate stress level
    current_stress = calculate_stress_level(emotion_avg, eye_contact_avg, head_movement_avg)
    st.session_state.stress_history.append(current_stress)
    stress_avg = sum(st.session_state.stress_history) / max(1, len(st.session_state.stress_history))
    stress_label, stress_emoji = get_stress_label(stress_avg)

    # Determine overall eye contact performance
    if eye_contact_avg >= 80:
        eye_contact_perf = "Good"
    elif eye_contact_avg >= 40:
        eye_contact_perf = "Average"
    else:
        eye_contact_perf = "Poor"

    return emotion_avg, eye_contact_avg, head_movement_avg, stress_avg, stress_label, stress_emoji, eye_contact_perf

# =========================
# UI
# =========================
st.set_page_config(page_title="Real-time Emotion, Eye Contact & Stress Detection", page_icon="😊", layout="wide")
st.title("😊 Real-time Emotion, Eye Contact & Stress Detection")
st.write("Webcam captures emotions, eye contact, head movement, and calculates stress levels using behavioral analysis.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera", disabled=st.session_state.camera_active):
        st.session_state.camera_active = True
        st.session_state.history.clear()
        st.session_state.eye_contact_history.clear()
        st.session_state.head_movement_history.clear()
        st.session_state.stress_history.clear()
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
        stress_placeholder = st.empty()

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
                emotion_avg, eye_contact_avg, head_movement_avg, stress_avg, stress_label, stress_emoji, eye_contact_perf = update_rollups(result)

                # Display current metrics
                # current_emotion = map_emotion(result.get("emotion","None"))  # Commented out mapping
                current_emotion = result.get("emotion", "None")  # Use original emotion
                confidence = result.get("confidence", 0)
                eye_contact = result.get("eye_contact", 0)
                head_movement = result.get("head_movement", 0)
                
                status_placeholder.success(
                    f"Emotion: **{current_emotion}** ({confidence}%) | "
                    f"Eye Contact: **{eye_contact}%** | "
                    f"Head Movement: **{head_movement}%**"
                )

                # Display stress level
                stress_placeholder.markdown(
                    f"### {stress_emoji} Stress Level: {stress_label} ({stress_avg:.1f}/100)"
                )

                # Display rolling averages
                sorted_emotion_avg = sorted(emotion_avg.items(), key=lambda kv: kv[1], reverse=True)
                details_text = "**Average Emotion Percentages:**\n\n"
                for e, v in sorted_emotion_avg:
                    details_text += f"- {e}: {round(v*100,1)}%\n"
                
                details_text += f"\n**Behavioral Indicators:**\n"
                details_text += f"- Eye Contact: {round(eye_contact_avg,1)}% ({eye_contact_perf})\n"
                details_text += f"- Head Movement: {round(head_movement_avg,1)}%\n"
                details_text += f"- Calculated Stress: {stress_avg:.1f}/100 ({stress_label})"
                
                details_placeholder.markdown(details_text)

                # Bar chart
                chart_data = {k:[v] for k,v in emotion_avg.items()}
                chart_placeholder.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Backend error: {e}")

else:
    st.info("Click 'Start Camera' to begin detection")

# Reset button
if st.button("Reset All Metrics"):
    st.session_state.history.clear()
    st.session_state.eye_contact_history.clear()
    st.session_state.head_movement_history.clear()
    st.session_state.stress_history.clear()
    st.success("All metrics cleared")