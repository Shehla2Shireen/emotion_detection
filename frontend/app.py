import streamlit as st
import requests
from PIL import Image

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"
# API_URL = "https://emotion-detection-612403936271.us-central1.run.app"

st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")

st.title("😊 Emotion Detection App")
st.write("Upload a face image and the model will predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send file to FastAPI backend
    if st.button("Predict Emotion"):
        with st.spinner("Analyzing..."):
            try:
                # Ensure file pointer at start & read bytes
                uploaded_file.seek(0)
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "image/jpeg"
                    )
                }

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Predicted Emotion: {result['emotion']} 🎯")
                    st.write(f"Confidence: {result['confidence']}%")

                    st.subheader("All Predictions:")
                    st.json(result["all_predictions"])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")


# import streamlit as st
# import cv2
# import requests
# import time
# import numpy as np

# API_URL = "http://127.0.0.1:8000/predict"

# st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="😊", layout="centered")
# st.title("😊 Real-Time Emotion Detection")

# # State to track video streaming
# if "run_camera" not in st.session_state:
#     st.session_state.run_camera = False

# # Buttons
# col1, col2 = st.columns(2)
# if col1.button("▶️ Start Video"):
#     st.session_state.run_camera = True
# if col2.button("⏹ Stop Video"):
#     st.session_state.run_camera = False

# # Placeholder for video + results
# frame_placeholder = st.empty()
# result_placeholder = st.empty()

# # Start capturing when button pressed
# if st.session_state.run_camera:
#     cap = cv2.VideoCapture(0)  # 0 = default webcam

#     while st.session_state.run_camera:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to access webcam")
#             break

#         # Convert frame to JPEG bytes
#         _, buffer = cv2.imencode(".jpg", frame)
#         img_bytes = buffer.tobytes()

#         try:
#             files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
#             response = requests.post(API_URL, files=files)

#             if response.status_code == 200:
#                 result = response.json()
#                 label = f"Predicted Emotion: {result['emotion']} 🎯 ({result['confidence']}%)"
#                 result_placeholder.success(label)
#             else:
#                 result_placeholder.error(f"Error: {response.status_code}")

#         except Exception as e:
#             result_placeholder.error(f"Request failed: {e}")

#         # Show live video frame in Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

#         time.sleep(2)  # wait 2 seconds before next frame

#     cap.release()
#     cv2.destroyAllWindows()
