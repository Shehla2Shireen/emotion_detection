# InterviewSense AI

A comprehensive AI-powered interview analysis tool that provides
real-time feedback on candidate performance by analyzing emotions, eye
contact, and stress levels during interviews.

## Overview

InterviewSense AI combines computer vision and machine learning to
evaluate interview performance through:

-   **Real-time emotion detection** (Angry, Happy, Sad, Surprise,
    Neutral, Disgust, Fear)
-   **Eye contact tracking** with percentage-based scoring
-   **Stress level analysis** based on negative emotion dominance
-   **Comprehensive reporting** with actionable feedback

## Architecture

The system consists of two main components:

### Frontend (Streamlit Application)

-   Real-time webcam feed processing
-   Live dashboard with metrics visualization
-   Video upload and analysis capability
-   Admin configuration panel
-   Automated report generation in Word format

### Backend (FastAPI Server)

-   TensorFlow emotion detection model
-   MediaPipe for facial landmark detection
-   Eye contact estimation algorithms
-   Head movement tracking
-   REST API endpoint for image processing

## Key Features

### Multi-Page Interface:

-   **Interview Dashboard**: Real-time webcam analysis
-   **Video Upload**: Process recorded interviews
-   **Admin Dashboard**: Configure evaluation parameters

### Smart Analysis:

-   15-frame rolling average for smooth metrics
-   Emotion dominance tracking (not just averages)
-   Configurable expected ranges for different emotions
-   Binary stress classification (High/Low) based on negative emotion
    prevalence

### Professional Reporting:

-   Automated Microsoft Word report generation
-   Color-coded performance indicators
-   Detailed metrics and compliance tables
-   Personalized recommendations for improvement

## Installation & Setup

### Prerequisites

-   Python 3.8+
-   Webcam (for live analysis)
-   Stable internet connection (for model download)

### Backend Setup

``` bash
# Install Python dependencies
pip install fastapi uvicorn tensorflow mediapipe opencv-python pillow python-multipart gdown numpy

# Start the backend server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

``` bash
# Install Python dependencies
pip install streamlit opencv-python requests python-docx

# Launch the Streamlit application
streamlit run app.py
```

## Usage

1.  Start the backend server on port 8000\
2.  Launch the frontend application\
3.  Navigate between:
    -   **Interview Dashboard**: For live webcam analysis\
    -   **Video Upload**: For processing recorded interviews\
    -   **Admin Dashboard**: To configure evaluation parameters\
4.  View real-time metrics including:
    -   Current dominant emotion\
    -   Eye contact percentage\
    -   Stress level classification\
    -   Emotion distribution compliance\
5.  Download comprehensive reports in Word format after session
    completion

## Configuration

The Admin Dashboard allows customization of: - Expected emotion ranges
(Neutral, Happy, Surprise, Negatives) - Ideal performance thresholds -
Evaluation parameters

## Technical Details

-   **Emotion Model**: Pre-trained TensorFlow model downloaded from
    Google Drive
-   **Face Detection**: Haar Cascade classifier
-   **Eye Tracking**: MediaPipe Face Mesh for precise iris detection
-   **Stress Calculation**: Based on percentage of frames where negative
    emotions are dominant
-   **Data Persistence**: JSON-based configuration saving

## API Endpoints

-   `POST /predict`: Accepts image files and returns emotion
    predictions, eye contact percentage, and head movement metrics

## Output Metrics

-   **Emotion Analysis**: Confidence scores for all 7 emotions
-   **Eye Contact**: Percentage score (0-100%) with performance
    categorization
-   **Stress Level**: Binary classification (High/Low) based on negative
    emotion prevalence
-   **Overall Interview Status**: Composite score (Good/Average/Bad)
    based on multiple factors

## File Structure

    interviewsense-ai/
    ├── app.py                 # Streamlit frontend application
    ├── main.py                # FastAPI backend server
    ├── admin_settings.json    # Configuration storage
    ├── haarcascade_frontalface_default.xml  # Face detection model
    └── artifacts/
        └── model-hopeful-meadow-1v12/  # Emotion detection model

## Troubleshooting

-   **Webcam not working**: Check camera permissions and ensure no other
    application is using the camera\
-   **Backend connection error**: Verify the backend server is running
    on port 8000\
-   **Model download issues**: Check internet connection and verify
    Google Drive file accessibility\
-   **Performance issues**: Reduce video resolution or close other
    resource-intensive applications

## Limitations

-   Requires good lighting conditions for accurate analysis
-   Works best with front-facing camera positions
-   May have reduced accuracy with glasses or extreme facial expressions
-   Performance depends on hardware capabilities

## Future Enhancements

-   Multi-person interview analysis
-   Audio sentiment analysis integration
-   Advanced analytics and trend tracking
-   Cloud-based processing for improved performance
-   Mobile application version

------------------------------------------------------------------------

This tool is designed for HR professionals, interview coaches, and
candidates looking to improve their interview performance through
data-driven feedback.
