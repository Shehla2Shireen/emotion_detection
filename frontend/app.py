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
API_URL = "https://emotion-backend-334182526116.us-central1.run.app/predict"
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

# =========================
# HELPERS
# =========================
def map_emotion(emotion: str) -> str:
    return MAP_TO_SAD.get(emotion, emotion)

def post_to_backend(img_bgr: np.ndarray):
    _, buf = cv2.imencode(".jpg", img_bgr)
    files = {"file": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    r = requests.post(API_URL, files=files, timeout=30)
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
st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")
st.title("😊 Emotion Detection App")
st.write("Upload an image or video. Emotions are shown alongside media in real-time.")
st.markdown("---")

# -------------------------
# IMAGE UPLOAD
# -------------------------
st.subheader("Upload an Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="img_upload")

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    img_array = np.array(image)[:, :, ::-1]  # PIL to BGR

    col1, col2 = st.columns([2,1])
    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    try:
        result = post_to_backend(img_array)
        result['emotion'] = map_emotion(result['emotion'])  # remap current emotion
        avg = update_rollups(result)
        with col2:
            st.success(f"Current: **{result['emotion']}** ({result['confidence']}%)")
            sorted_avg = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
            st.write(f"Rolling avg (last {len(st.session_state.history)} frames): **{sorted_avg[0][0]}** ({round(sorted_avg[0][1]*100,1)}%)")
            st.bar_chart({k:[v] for k,v in avg.items()})
    except Exception as e:
        with col2:
            st.error(f"Request failed: {e}")

st.markdown("---")

# -------------------------
# VIDEO UPLOAD
# -------------------------
st.subheader("Upload a Video")
uploaded_video = st.file_uploader("Choose a video...", type=["mp4","avi","mov"], key="vid_upload")

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    col1, col2 = st.columns([2,1])
    video_ph = col1.empty()
    current_ph = col2.empty()
    avg_ph = col2.empty()
    chart_ph = col2.empty()
    error_ph = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay current emotion on video
        now = time.time()
        if now - st.session_state.last_send_ts >= SEND_INTERVAL_SEC:
            try:
                result = post_to_backend(frame)
                result['emotion'] = map_emotion(result['emotion'])
                avg = update_rollups(result)
                st.session_state.last_send_ts = now
                emotion_text = f"{result['emotion']} ({result['confidence']}%)"

                current_ph.success(f"Current: **{result['emotion']}** ({result['confidence']}%)")
                sorted_avg = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
                avg_ph.write(f"Rolling avg (last {len(st.session_state.history)} frames): **{sorted_avg[0][0]}** ({round(sorted_avg[0][1]*100,1)}%)")
                chart_ph.bar_chart({k:[v] for k,v in avg.items()})

            except Exception as e:
                error_ph.error(f"Request failed: {e}")
                emotion_text = ""

        if 'emotion_text' in locals() and emotion_text:
            cv2.putText(frame, emotion_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=500)
        time.sleep(0.05)

    cap.release()

# =========================
# RESET BUTTON
# =========================
if st.button("Reset rolling average"):
    st.session_state.history.clear()
    video_ph.empty()
    current_ph.empty()
    avg_ph.empty()
    chart_ph.empty()
    st.info("Averages cleared.")
