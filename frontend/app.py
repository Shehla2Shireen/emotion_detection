import streamlit as st
import cv2
import io
import numpy as np
import requests
from collections import deque, defaultdict
import json
import os

# =========================
# CONFIG
# =========================
BACKEND_URL = "http://localhost:8000/predict"
ROLLING_WINDOW = 15
FPS = 15
SETTINGS_FILE = "admin_settings.json"

EMOTIONS = ['Angry','Happy','Sad','Surprise','Neutral','Disgust','Fear']

# Ideal defaults (for reset)
IDEAL_EMOTION_WEIGHTS = {
    'Angry': 0.29,
    'Sad': 0.25,
    'Fear': 0.33,
    'Disgust': 0.13,
    # Non-negative emotions get weight 0 (ignored in stress calc)
    'Happy': 0.0,
    'Surprise': 0.0,
    'Neutral': 0.0
}
IDEAL_STRESS_WEIGHTS = {'emotions': 0.7, 'eye_contact': 0.3}  # sum=1
IDEAL_EXPECTED_RANGES = {
    "Neutral": (0.50, 0.70),
    "Happy": (0.20, 0.30),
    "Surprise": (0.05, 0.10),
    "Negatives": (0.00, 0.10)
}

# =========================
# SETTINGS PERSISTENCE
# =========================
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {
        "EMOTION_WEIGHTS": IDEAL_EMOTION_WEIGHTS,
        "STRESS_WEIGHTS": IDEAL_STRESS_WEIGHTS,
        "EXPECTED_RANGES": IDEAL_EXPECTED_RANGES
    }

def save_settings():
    settings = {
        "EMOTION_WEIGHTS": st.session_state.EMOTION_WEIGHTS,
        "STRESS_WEIGHTS": st.session_state.STRESS_WEIGHTS,
        "EXPECTED_RANGES": st.session_state.EXPECTED_RANGES
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# =========================
# SESSION STATE DEFAULTS
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
if "smoothed_stress" not in st.session_state:
    st.session_state.smoothed_stress = 0

# Load settings into session state
loaded_settings = load_settings()
if "EMOTION_WEIGHTS" not in st.session_state:
    st.session_state.EMOTION_WEIGHTS = loaded_settings["EMOTION_WEIGHTS"]
if "STRESS_WEIGHTS" not in st.session_state:
    st.session_state.STRESS_WEIGHTS = loaded_settings["STRESS_WEIGHTS"]
if "EXPECTED_RANGES" not in st.session_state:
    st.session_state.EXPECTED_RANGES = loaded_settings["EXPECTED_RANGES"]

ALPHA = 0.3

# =========================
# HELPERS
# =========================
def get_stress_label(stress_score: float):
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

def calculate_stress_level(emotions, eye_contact):
    # Emotion stress = only negative emotions with weights
    weighted_sum = 0
    for emo in ['Angry', 'Sad', 'Fear', 'Disgust']:
        weighted_sum += emotions.get(emo, 0) * st.session_state.EMOTION_WEIGHTS.get(emo, 0)
    emotion_component = min(100, weighted_sum * 100)

    # Eye contact stress
    eye_contact_component = min(100, (100 - eye_contact))

    raw_stress = (
        st.session_state.STRESS_WEIGHTS['emotions'] * emotion_component +
        st.session_state.STRESS_WEIGHTS['eye_contact'] * eye_contact_component
    )
    smoothed = ALPHA * raw_stress + (1 - ALPHA) * st.session_state.smoothed_stress
    st.session_state.smoothed_stress = smoothed
    return smoothed

def update_rollups(emotion_result):
    probs = emotion_result.get("all_predictions", {})
    full = {e: 0.0 for e in EMOTIONS}
    for k, v in probs.items():
        full[k] += float(v)
    st.session_state.history.append(full)
    st.session_state.eye_contact_history.append(emotion_result.get("eye_contact", 0))
    st.session_state.head_movement_history.append(emotion_result.get("head_movement", 0))

    emotion_avg = defaultdict(float)
    for row in st.session_state.history:
        for k, v in row.items():
            emotion_avg[k] += v
    n = max(1, len(st.session_state.history))
    for k in emotion_avg:
        emotion_avg[k] /= n

    eye_contact_avg = sum(st.session_state.eye_contact_history) / max(1, len(st.session_state.eye_contact_history))
    head_movement_avg = sum(st.session_state.head_movement_history) / max(1, len(st.session_state.head_movement_history))

    current_stress = calculate_stress_level(emotion_avg, eye_contact_avg)
    st.session_state.stress_history.append(current_stress)
    stress_avg = sum(st.session_state.stress_history) / max(1, len(st.session_state.stress_history))
    stress_label, stress_emoji = get_stress_label(stress_avg)

    if eye_contact_avg >= 80:
        eye_contact_perf = "Good"
    elif eye_contact_avg >= 40:
        eye_contact_perf = "Average"
    else:
        eye_contact_perf = "Poor"

    return emotion_avg, eye_contact_avg, head_movement_avg, stress_avg, stress_label, stress_emoji, eye_contact_perf

def evaluate_emotion_distribution(emotion_avg):
    total = sum(emotion_avg.values())
    norm = {k: v/total for k,v in emotion_avg.items()} if total > 0 else emotion_avg
    
    report = {}
    for key, (low, high) in st.session_state.EXPECTED_RANGES.items():
        if key == "Negatives":
            neg_val = norm.get("Angry",0)+norm.get("Sad",0)+norm.get("Fear",0)+norm.get("Disgust",0)
            report[key] = (neg_val, low <= neg_val <= high)
        else:
            val = norm.get(key, 0)
            report[key] = (val, low <= val <= high)
    return report

# =========================
# MULTI-PAGE DASHBOARD
# =========================
st.set_page_config(page_title="Interview Analysis Tool", page_icon="🎥", layout="wide")
page = st.sidebar.radio("Navigate", ["Interview Dashboard", "Admin Dashboard"])

# =========================
# PAGE 1: INTERVIEW DASHBOARD
# =========================
if page == "Interview Dashboard":
    st.title("🎥 Interview Dashboard")
    st.caption("Real-time detection of emotions, eye contact, head movement, and stress.")

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

            left_col, right_col = st.columns([2,2])
            with left_col:
                video_placeholder = st.empty()
            with right_col:
                status_placeholder = st.empty()
                stress_placeholder = st.empty()
                details_placeholder = st.empty()
                evaluation_placeholder = st.empty()
                chart_placeholder = st.empty()

            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", width=500)

                try:
                    result = post_to_backend(frame)
                    (emotion_avg, eye_contact_avg, head_movement_avg,
                     stress_avg, stress_label, stress_emoji, eye_contact_perf) = update_rollups(result)

                    current_emotion = result.get("emotion", "None")
                    confidence = result.get("confidence", 0)
                    eye_contact = result.get("eye_contact", 0)
                    head_movement = result.get("head_movement", 0)
                    
                    status_placeholder.markdown(
                        f"**Emotion:** {current_emotion} ({confidence}%)  |  "
                        f"**Eye Contact:** {eye_contact}%  |  "
                        f"**Head Move.:** {head_movement}%"
                    )
                    stress_placeholder.markdown(
                        f"**{stress_emoji} Stress:** {stress_label} ({stress_avg:.1f}/100)"
                    )

                    sorted_emotion_avg = sorted(emotion_avg.items(), key=lambda kv: kv[1], reverse=True)
                    details_text = "📊 **Averages:**  " + "  |  ".join([f"{e}: {round(v*100,1)}%" for e,v in sorted_emotion_avg])
                    details_text += f"\n\n👀 **Eye Contact:** {round(eye_contact_avg,1)}% ({eye_contact_perf})"
                    details_text += f"  |  🤷 **Head Move.:** {round(head_movement_avg,1)}%"
                    details_text += f"  |  🔎 **Stress:** {stress_avg:.1f}/100 ({stress_label})"
                    details_placeholder.markdown(details_text)

                    evaluation_report = evaluate_emotion_distribution(emotion_avg)
                    eval_text = "🎯 **Emotion Ranges:**\n"
                    for k, (val, ok) in evaluation_report.items():
                        percent = round(val*100,1)
                        low, high = st.session_state.EXPECTED_RANGES[k]
                        status = "✅" if ok else "⚠️"
                        eval_text += f"- {k}: {percent}% (exp. {int(low*100)}–{int(high*100)}%) {status}\n"
                    evaluation_placeholder.markdown(eval_text)

                    chart_data = {k:[v] for k,v in emotion_avg.items()}
                    chart_placeholder.bar_chart(chart_data)

                except Exception as e:
                    st.error(f"Backend error: {e}")

    else:
        st.info("Click 'Start Camera' to begin detection")

    if st.button("Reset All Metrics"):
        st.session_state.history.clear()
        st.session_state.eye_contact_history.clear()
        st.session_state.head_movement_history.clear()
        st.session_state.stress_history.clear()
        st.success("All metrics cleared")

# =========================
# PAGE 2: ADMIN DASHBOARD
# =========================
if page == "Admin Dashboard":
    st.title("⚙️ Admin Dashboard")
    st.caption("Configure weights and expected ranges for interview evaluation.")

    st.subheader("📌 Negative Emotion Weights (for stress calculation)")
    for emo in ['Angry','Sad','Fear','Disgust']:
        col1, col2 = st.columns([4,1])
        with col1:
            st.session_state.EMOTION_WEIGHTS[emo] = st.slider(
                f"{emo} weight", 0.0, 2.0, st.session_state.EMOTION_WEIGHTS.get(emo, IDEAL_EMOTION_WEIGHTS[emo]), 0.1, key=f"emo_{emo}"
            )
        with col2:
            if st.button("Reset", key=f"reset_{emo}"):
                st.session_state.EMOTION_WEIGHTS[emo] = IDEAL_EMOTION_WEIGHTS[emo]
                save_settings()
                st.rerun()

    st.subheader("📌 Stress Component Weights")
    for comp in ["emotions","eye_contact"]:
        col1, col2 = st.columns([4,1])
        with col1:
            st.session_state.STRESS_WEIGHTS[comp] = st.slider(
                f"{comp} weight", 0.0, 1.0, st.session_state.STRESS_WEIGHTS[comp], 0.05, key=f"comp_{comp}"
            )
        with col2:
            if st.button("Reset", key=f"reset_{comp}"):
                st.session_state.STRESS_WEIGHTS[comp] = IDEAL_STRESS_WEIGHTS[comp]
                save_settings()
                st.rerun()
    # Normalize
    s = sum(st.session_state.STRESS_WEIGHTS.values())
    for k in st.session_state.STRESS_WEIGHTS:
        st.session_state.STRESS_WEIGHTS[k] /= s

    st.subheader("📌 Expected Emotion Ranges")
    for emo in st.session_state.EXPECTED_RANGES.keys():
        col1, col2 = st.columns([4,1])
        with col1:
            low, high = st.session_state.EXPECTED_RANGES[emo]
            st.session_state.EXPECTED_RANGES[emo] = st.slider(
                f"{emo} % range", 0.0, 1.0, (low, high), 0.05, key=f"range_{emo}"
            )
        with col2:
            if st.button("Reset", key=f"reset_range_{emo}"):
                st.session_state.EXPECTED_RANGES[emo] = IDEAL_EXPECTED_RANGES[emo]
                save_settings()
                st.rerun()

    # Always save latest settings
    save_settings()
    st.success("✅ Admin settings saved. They will apply in Interview Dashboard.")
