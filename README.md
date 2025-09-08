# 🎭 Facial Emotion & Stress Detection System
**AI-powered interview analysis tool using Deep Learning, FastAPI, Streamlit, and Weights & Biases (W&B)**  

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![WandB](https://img.shields.io/badge/Weights&Biases-Tracking-yellow)

---

## 📌 Overview
This project detects **7 basic facial emotions** in real-time and calculates a **stress level** based on emotional patterns.
It can be used for **interview evaluation**, **workplace productivity monitoring**, and **HR analytics**.

---

## 🚀 Features
✅ Real-time emotion detection from webcam or uploaded videos  
✅ Supports **7 emotions** → `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`  
✅ Stress level estimation based on detected emotions  
✅ W&B artifact integration for model loading  
✅ Streamlit dashboard for interview reports  
✅ REST API using FastAPI for integration with other apps  

---

## 🧠 Model Details
- **Architecture** → CNN + Dense layers  
- **Input Shape** → `48x48x1` grayscale images  
- **Output Classes** → 7 emotions  
- **Training Dataset** → FER-2013  
- **Tracked with** → [Weights & Biases](https://wandb.ai/)

**Loading the Model from W&B Artifact**:
```python
import wandb
import tensorflow as tf

wandb.login()
artifact = wandb.use_artifact(
    'shehlashireen03-atomcamp/emotion_detection/model-hopeful-meadow-1:v12', 
    type='model'
)
artifact_dir = artifact.download()
model = tf.keras.models.load_model(artifact_dir)
```

---

## 📊 Stress Level Calculation
**Formula**:
\[
Stress Level = \frac{(1.0*Angry + 0.9*Fear + 0.8*Sad + 0.5*Disgust)}{(1.2*Happy + 0.8*Neutral + 1)} \times 100
\]

**Reference:** Inspired by  
*"Stress Detection through Compound Facial Expressions Using Neural Networks"*  
[DOI: 10.55549/epess.807](https://doi.org/10.55549/epess.807)

---

## 🖥️ Running the Project

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the FastAPI Backend**
```bash
cd backend
uvicorn app:app --reload
```
**API Endpoint** → `http://127.0.0.1:8000/predict`

### **3️⃣ Run the Streamlit Dashboard**
```bash
cd dashboard
streamlit run streamlit_app.py
```
**Dashboard URL** → `http://localhost:8501`

### **4️⃣ Predict Emotions via API**
```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {'file': open("sample_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 🛠️ Tech Stack
- **Backend** → FastAPI
- **Frontend** → Streamlit
- **Model Tracking** → Weights & Biases
- **Deep Learning** → TensorFlow / Keras
- **Database (optional)** → PostgreSQL / MongoDB
- **Deployment** → Docker + Cloud-ready

---

## 📌 Future Enhancements
- ✅ Integrate **voice tone analysis** for better stress detection  
- ✅ Multi-user dashboard for HR analytics  
- ✅ Real-time analytics for group interviews  
- ✅ Optimize inference speed using TensorRT  

