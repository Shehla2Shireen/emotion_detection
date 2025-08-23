import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="😊", layout="centered")
st.title("😊 Real-Time Emotion Detection")

# Start/Stop camera button
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

col1, col2 = st.columns(2)
if col1.button("▶️ Start Video"):
    st.session_state.run_camera = True
if col2.button("⏹ Stop Video"):
    st.session_state.run_camera = False

# Placeholders
result_placeholder = st.empty()

if st.session_state.run_camera:
    camera_frame = st.camera_input("Webcam feed")

    if camera_frame is not None:
        # Convert to bytes
        img_bytes = camera_frame.getvalue()

        try:
            files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Emotion: {result['emotion']} 🎯 ({result['confidence']}%)")
                st.json(result["all_predictions"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")
