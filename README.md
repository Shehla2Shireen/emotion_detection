# Emotion Detection — End-to-End (TF/Keras + FastAPI + Streamlit + W&B)

This project turns your notebook into a production-ready pipeline:

- **Training**: TensorFlow/Keras on FER2013 (CSV). W&B tracks runs and saves model artifacts.
- **Backend**: FastAPI API for inference (accepts image upload or base64).
- **Frontend**: Streamlit UI to upload an image and view prediction by calling the API.
- **Deployment**: Ready for Google Cloud Run (backend). Streamlit can be deployed on Streamlit Cloud or Cloud Run.

## Quickstart (Local)

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
# Or: source .venv/bin/activate   # macOS/Linux

pip install -r training/requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 1) Train (logs to W&B; saves model as artifact)
python training/train.py --csv_path path/to/fer2013.csv --project ml-end-to-end --run_name baseline-v1

# 2) Run API
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3) Run UI
streamlit run frontend/app.py
```

### Environment

- Python 3.10+ recommended
- Set W&B once: `wandb login`

## Deploy to Google Cloud Run (Backend)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/emotion-api ./backend
gcloud run deploy emotion-api --image gcr.io/PROJECT_ID/emotion-api --platform managed --allow-unauthenticated --region YOUR_REGION
```

Update `frontend/app.py` to point to the deployed API URL.

## Videeo Link
Here is the link to demonstration video
https://drive.google.com/file/d/1PJos6Xp5AD_RpzDDJsub0w7N_pqkZ10S/view?usp=sharing

