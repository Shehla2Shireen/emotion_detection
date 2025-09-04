import streamlit as st
import cv2
import io
import numpy as np
import requests
from collections import deque, defaultdict
import json
import os
from docx import Document
from docx.shared import Inches
from docx.shared import RGBColor

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
    # Non-negative emotions ignored in stress calc
    'Happy': 0.0,
    'Surprise': 0.0,
    'Neutral': 0.0
}
IDEAL_STRESS_WEIGHTS = {'emotions': 0.7, 'eye_contact': 0.3}  # unused now
IDEAL_EXPECTED_RANGES = {
    "Neutral": (0.50, 0.70),
    "Happy": (0.01, 0.05),
    "Surprise": (0.01, 0.05),
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
if "eye_contact_binary_history" not in st.session_state:
    st.session_state.eye_contact_binary_history = deque(maxlen=ROLLING_WINDOW)
if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque(maxlen=ROLLING_WINDOW)
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Load settings into session state
loaded_settings = load_settings()
if "EMOTION_WEIGHTS" not in st.session_state:
    st.session_state.EMOTION_WEIGHTS = loaded_settings["EMOTION_WEIGHTS"]
if "STRESS_WEIGHTS" not in st.session_state:
    st.session_state.STRESS_WEIGHTS = loaded_settings["STRESS_WEIGHTS"]
if "EXPECTED_RANGES" not in st.session_state:
    st.session_state.EXPECTED_RANGES = loaded_settings["EXPECTED_RANGES"]

# =========================
# HELPERS
# =========================
def get_stress_label(evaluation_report: dict):
    neg_val, ok = evaluation_report.get("Negatives", (0, True))
    if ok:
        return "Low Stress", "🟢"
    else:
        return "High Stress", "🔴"

def get_interview_status(stress_label, eye_contact_percentage, evaluation_report):
    stress_score_val = 100 if stress_label == "Low Stress" else 40  # simple mapping
    eye_contact_score = eye_contact_percentage
    compliance_score = 0
    total_ranges = len(evaluation_report)
    for _, (_, ok) in evaluation_report.items():
        compliance_score += 100 if ok else 60
    compliance_score /= total_ranges
    overall_score = (
        stress_score_val * 0.4 + 
        eye_contact_score * 0.3 + 
        compliance_score * 0.3
    )
    if overall_score >= 80:
        return "Good ✅", overall_score, {
            "stress_score": stress_score_val,
            "eye_contact_score": eye_contact_score,
            "emotion_compliance_score": compliance_score
        }
    elif overall_score >= 60:
        return "Average ⚠️", overall_score, {
            "stress_score": stress_score_val,
            "eye_contact_score": eye_contact_score,
            "emotion_compliance_score": compliance_score
        }
    else:
        return "Bad ❌", overall_score, {
            "stress_score": stress_score_val,
            "eye_contact_score": eye_contact_score,
            "emotion_compliance_score": compliance_score
        }

def post_to_backend(img_bgr: np.ndarray):
    _, buf = cv2.imencode(".jpg", img_bgr)
    files = {"file": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    r = requests.post(BACKEND_URL, files=files, timeout=30)
    r.raise_for_status()
    return r.json()

def update_rollups(emotion_result):
    probs = emotion_result.get("all_predictions", {})
    full = {e: 0.0 for e in EMOTIONS}
    for k, v in probs.items():
        full[k] += float(v)
    st.session_state.history.append(full)

    # --- Eye contact handling ---
    eye_contact_value = emotion_result.get("eye_contact", 0)
    eye_contact_binary = 100 if eye_contact_value >= 95 else 0
    st.session_state.eye_contact_history.append(eye_contact_value)
    st.session_state.eye_contact_binary_history.append(eye_contact_binary)

    # --- Average probabilities (not used for ranges anymore, but kept for charts) ---
    emotion_avg = defaultdict(float)
    for row in st.session_state.history:
        for k, v in row.items():
            emotion_avg[k] += v
    n = max(1, len(st.session_state.history))
    for k in emotion_avg:
        emotion_avg[k] /= n

    # --- Eye contact metrics ---
    eye_contact_percentage = sum(st.session_state.eye_contact_binary_history) / max(1, len(st.session_state.eye_contact_binary_history))
    eye_contact_avg = sum(st.session_state.eye_contact_history) / max(1, len(st.session_state.eye_contact_history))

    # --- Track dominant emotions history ---
    if "dominance_history" not in st.session_state:
        st.session_state.dominance_history = deque(maxlen=5000)  # big enough for whole session
    dominant_emotion = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
    if dominant_emotion:
        st.session_state.dominance_history.append(dominant_emotion)

    # --- Build evaluation report from dominance percentages ---
    evaluation_report = evaluate_emotion_distribution(emotion_avg)

    # --- Stress label ---
    stress_label, stress_emoji = get_stress_label(evaluation_report)
    st.session_state.stress_history.append(1 if stress_label == "High Stress" else 0)

    # --- Eye contact performance bucket ---
    if eye_contact_percentage >= 80:
        eye_contact_perf = "Good"
    elif eye_contact_percentage >= 40:
        eye_contact_perf = "Average"
    else:
        eye_contact_perf = "Poor"

    return emotion_avg, eye_contact_avg, eye_contact_percentage, stress_label, stress_emoji, eye_contact_perf, dominant_emotion, evaluation_report


def evaluate_emotion_distribution(emotion_avg):
    """
    Instead of using normalized average probabilities,
    compute percentage of frames where each emotion was dominant
    and compare with expected ranges.
    """
    if "dominance_history" not in st.session_state or len(st.session_state.dominance_history) == 0:
        return {k: (0.0, False) for k in st.session_state.EXPECTED_RANGES}

    dominance_counts = {emo: 0 for emo in EMOTIONS}
    for emo in st.session_state.dominance_history:
        dominance_counts[emo] += 1
    total_frames = len(st.session_state.dominance_history)
    dominance_percentages = {emo: dominance_counts[emo] / total_frames for emo in EMOTIONS}

    report = {}
    for key, (low, high) in st.session_state.EXPECTED_RANGES.items():
        if key == "Negatives":
            neg_val = (
                dominance_percentages.get("Angry", 0)
                + dominance_percentages.get("Sad", 0)
                + dominance_percentages.get("Fear", 0)
                + dominance_percentages.get("Disgust", 0)
            )
            report[key] = (neg_val, low <= neg_val <= high)
        else:
            val = dominance_percentages.get(key, 0)
            report[key] = (val, low <= val <= high)
    return report


# =========================
# REPORT GENERATION (unchanged except stress mapping)
# =========================
def add_colored_text(paragraph, text, color):
    run = paragraph.add_run(text)
    if color == "green":
        run.font.color.rgb = RGBColor(0, 128, 0)
    elif color == "red":
        run.font.color.rgb = RGBColor(220, 20, 60)
    elif color == "orange":
        run.font.color.rgb = RGBColor(255, 140, 0)
    return paragraph


def generate_report(emotion_avg, eye_contact_avg, eye_contact_percentage, stress_label, evaluation_report):
    doc = Document()
    doc.add_heading("Interview Analysis Report", 0)
    
    # Executive Summary
    doc.add_heading("Executive Summary", level=1)
    status_para = doc.add_paragraph()
    status_para.add_run("Overall Interview Stress: ")
    if stress_label == "Low Stress":
        add_colored_text(status_para, stress_label, "green")
    else:
        add_colored_text(status_para, stress_label, "red")
    
    # Ideal parameters reference
    doc.add_heading("Ideal Parameters Reference", level=2)
    ideal_para = doc.add_paragraph()
    ideal_para.add_run("• Stress Level: ").bold = True
    ideal_para.add_run("Low Stress (emotions within expected range)\n")
    ideal_para.add_run("• Eye Contact: ").bold = True
    ideal_para.add_run("70-100% of time with good eye contact (Higher is better)\n")
    ideal_para.add_run("• Emotion Ranges: ").bold = True
    ideal_para.add_run("Neutral (50-70%), Happy (20-30%), Surprise (5-10%), Negative Emotions (0-10%)")
    
    # Detailed Metrics
    doc.add_heading("Detailed Metrics", level=1)
    metrics_table = doc.add_table(rows=3, cols=4)
    metrics_table.style = 'Table Grid'
    
    # Table headers
    metrics_table.cell(0, 0).text = "Metric"
    metrics_table.cell(0, 1).text = "Value"
    metrics_table.cell(0, 2).text = "Ideal Range"
    metrics_table.cell(0, 3).text = "Status"
    
    # Stress (binary now)
    metrics_table.cell(1, 0).text = "Stress Level"
    metrics_table.cell(1, 1).text = stress_label
    metrics_table.cell(1, 2).text = "Low Stress"
    stress_cell = metrics_table.cell(1, 3)
    if stress_label == "Low Stress":
        stress_cell.text = "Good ✅"
        for paragraph in stress_cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0, 128, 0)
    else:
        stress_cell.text = "Poor ❌"
        for paragraph in stress_cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(220, 20, 60)
    
    # Eye Contact (Percentage of time with good eye contact)
    metrics_table.cell(2, 0).text = "Eye Contact Time"
    metrics_table.cell(2, 1).text = f"{eye_contact_percentage:.1f}%"
    metrics_table.cell(2, 2).text = "70-100%"
    eye_contact_cell = metrics_table.cell(2, 3)
    if eye_contact_percentage >= 70:
        eye_contact_cell.text = "Good ✅"
        for paragraph in eye_contact_cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0, 128, 0)
    elif eye_contact_percentage >= 40:
        eye_contact_cell.text = "Average ⚠️"
        for paragraph in eye_contact_cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(255, 140, 0)
    else:
        eye_contact_cell.text = "Poor ❌"
        for paragraph in eye_contact_cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(220, 20, 60)
    
    # =====================
    # Emotion Range Compliance ONLY
    # =====================
    doc.add_heading("Emotion Range Compliance", level=1)
    compliance_table = doc.add_table(rows=1, cols=4)
    compliance_table.style = 'Table Grid'
    
    hdr_cells = compliance_table.rows[0].cells
    hdr_cells[0].text = 'Category'
    hdr_cells[1].text = 'Observed %'
    hdr_cells[2].text = 'Expected Range'
    hdr_cells[3].text = 'Status'
    
    for emo, (val, ok) in evaluation_report.items():
        row_cells = compliance_table.add_row().cells
        row_cells[0].text = emo
        row_cells[1].text = f"{val*100:.1f}%"
        low, high = st.session_state.EXPECTED_RANGES[emo]
        row_cells[2].text = f"{low*100:.0f}–{high*100:.0f}%"
        status_cell = row_cells[3]
        if ok:
            status_cell.text = "✅ Within range"
            for paragraph in status_cell.paragraphs:
                for run in paragraph.runs:
                    run.font.color.rgb = RGBColor(0, 128, 0)
        else:
            status_cell.text = "⚠️ Outside range"
            for paragraph in status_cell.paragraphs:
                for run in paragraph.runs:
                    run.font.color.rgb = RGBColor(220, 20, 60)

    # Recommendations
    doc.add_heading("Key Recommendations", level=1)
    if stress_label == "High Stress":
        doc.add_paragraph("Candidate showed signs of high stress. Consider stress management coaching or interview practice in low-pressure settings.")
    else:
        doc.add_paragraph("Candidate maintained low stress levels, showing composure during the interview.")
    
    if eye_contact_percentage < 60:
        doc.add_paragraph("Eye contact needs improvement. Practice maintaining eye contact for longer periods.")
    elif eye_contact_percentage < 70:
        doc.add_paragraph("Eye contact is adequate but could be improved. Aim for ≥70%.")
    
    if any(not ok for _, ok in evaluation_report.items()):
        doc.add_paragraph("Emotional expression needs adjustment. Work on maintaining neutral to positive expressions during professional interactions.")
    
    filepath = "Interview_Report.docx"
    doc.save(filepath)
    return filepath


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
    st.caption("Real-time detection of emotions, eye contact, and stress.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera", disabled=st.session_state.camera_active):
            st.session_state.camera_active = True
            st.session_state.history.clear()
            st.session_state.eye_contact_history.clear()
            st.session_state.eye_contact_binary_history.clear()
            st.session_state.stress_history.clear()
            st.rerun()
    with col2:
        if st.button("Stop Camera", disabled=not st.session_state.camera_active):
            st.session_state.camera_active = False
            if len(st.session_state.history) > 0:
                emotion_avg = defaultdict(float)
                for row in st.session_state.history:
                    for k, v in row.items():
                        emotion_avg[k] += v
                n = max(1, len(st.session_state.history))
                for k in emotion_avg:
                    emotion_avg[k] /= n
                eye_contact_avg = sum(st.session_state.eye_contact_history)/max(1,len(st.session_state.eye_contact_history))
                eye_contact_percentage = sum(st.session_state.eye_contact_binary_history)/max(1,len(st.session_state.eye_contact_binary_history))
                evaluation_report = evaluate_emotion_distribution(emotion_avg)
                stress_label, _ = get_stress_label(evaluation_report)
                filepath = generate_report(emotion_avg, eye_contact_avg, eye_contact_percentage, stress_label, evaluation_report)


                with open(filepath, "rb") as f:
                    bytes_data = f.read()
                st.download_button(
                    label="⬇️ Interview Report (Auto Download)",
                    data=bytes_data,
                    file_name=filepath,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="auto_download"
                )

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
                chart_placeholder = st.empty()
                details_placeholder = st.empty()
                evaluation_placeholder = st.empty()

            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", width=500)

                try:
                    result = post_to_backend(frame)
                    (emotion_avg, eye_contact_avg, eye_contact_percentage, stress_label, stress_emoji, eye_contact_perf, dominant_emotion, evaluation_report) = update_rollups(result)
                    current_emotion = dominant_emotion  # make dominant emotion prominent
                    confidence = result.get("confidence", 0)
                    eye_contact = result.get("eye_contact", 0)
                    instant_emotions = result.get("all_predictions", {}) or {}
                    instant_sorted = sorted(instant_emotions.items(), key=lambda kv: kv[1], reverse=True)

                    if len(instant_sorted) > 0:
                        top_emo, top_val = instant_sorted[0]
                        status_placeholder.markdown(
                            f"<h2 style='text-align: center; color: {'green' if top_emo in ['Neutral','Happy'] else 'red'};'>"
                            f"🔥 Dominant Now: {top_emo} ({top_val*100:.1f}%)</h2>",
                            unsafe_allow_html=True
                        )
                        chart_placeholder.bar_chart({k:[v] for k,v in instant_emotions.items()})
                        instant_text = "📌 **Instantaneous Emotions (Current Frame):**\n"
                        for emo, val in instant_sorted:
                            instant_text += f"- {emo}: {val*100:.1f}%\n"
                        details_placeholder.markdown(instant_text)
                    else:
                        status_placeholder.markdown(
                            "<h2 style='text-align: center; color: orange;'>No face detected</h2>",
                            unsafe_allow_html=True
                        )
                        chart_placeholder.bar_chart({e:[0.0] for e in EMOTIONS})
                        details_placeholder.markdown("📌 **Instantaneous Emotions (Current Frame):**\n- No data for this frame.")

                    # Compute overall stress till now
                    overall_stress_ratio = sum(st.session_state.stress_history) / max(1, len(st.session_state.stress_history))
                    overall_stress_label = "High Stress" if overall_stress_ratio > 0.5 else "Low Stress"
                    overall_stress_emoji = "🔴" if overall_stress_label == "High Stress" else "🟢"

                    stress_placeholder.markdown(
                        f"<div style='text-align: center;'>"
                        f"<span style='font-size: 1.2em;'>👀 Eye Contact: {result.get('eye_contact', 0)}% (Current)</span> | "
                        f"<span style='font-size: 1.2em;'>⏱️ Good Eye Contact Time: {eye_contact_percentage:.1f}%</span> | "
                        f"<span style='font-size: 1.2em;'>{overall_stress_emoji} Stress (Overall): {overall_stress_label}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


                    sorted_emotion_avg = sorted(emotion_avg.items(), key=lambda kv: kv[1], reverse=True)
                    avg_text = "📊 **Smoothed Averages (15-frame window):**\n" + \
                            "  |  ".join([f"{e}: {round(v*100,1)}%" for e,v in sorted_emotion_avg])
                    avg_text += f"\n\n⏱️ **Good Eye Contact Time:** {eye_contact_percentage:.1f}% ({eye_contact_perf})"
                    avg_text += f"  |  🔎 **Stress:** {stress_label}"
                    evaluation_placeholder.markdown(avg_text)

                    eval_text = "🎯 **Emotion Ranges:**\n"
                    for k, (val, ok) in evaluation_report.items():
                        percent = round(val*100,1)
                        low, high = st.session_state.EXPECTED_RANGES[k]
                        status = "✅" if ok else "⚠️"
                        eval_text += f"- {k}: {percent}% (exp. {int(low*100)}–{int(high*100)}%) {status}\n"
                    evaluation_placeholder.markdown(eval_text)

                except Exception as e:
                    st.error(f"Backend error: {e}")

    else:
        st.info("Click 'Start Camera' to begin detection")


# =========================
# PAGE 2: ADMIN DASHBOARD
# =========================
if page == "Admin Dashboard":
    st.title("⚙️ Admin Dashboard")
    st.caption("Configure weights and expected ranges for interview evaluation.")

    # st.subheader("📌 Negative Emotion Weights (for stress calculation)")
    # for emo in ['Angry','Sad','Fear','Disgust']:
    #     col1, col2 = st.columns([4,1])
    #     with col1:
    #         st.session_state.EMOTION_WEIGHTS[emo] = st.slider(
    #             f"{emo} weight", 0.0, 2.0, st.session_state.EMOTION_WEIGHTS.get(emo, IDEAL_EMOTION_WEIGHTS[emo]), 0.1, key=f"emo_{emo}"
    #         )
    #     with col2:
    #         if st.button("Reset to Default ", key=f"reset_{emo}"):
    #             st.session_state.EMOTION_WEIGHTS[emo] = IDEAL_EMOTION_WEIGHTS[emo]
    #             save_settings()
    #             st.rerun()

    # st.subheader("📌 Stress Component Weights")
    # for comp in ["emotions","eye_contact"]:
    #     col1, col2 = st.columns([4,1])
    #     with col1:
    #         st.session_state.STRESS_WEIGHTS[comp] = st.slider(
    #             f"{comp} weight", 0.0, 1.0, st.session_state.STRESS_WEIGHTS[comp], 0.05, key=f"comp_{comp}"
    #         )
    #     with col2:
    #         if st.button("Reset to Default ", key=f"reset_{comp}"):
    #             st.session_state.STRESS_WEIGHTS[comp] = IDEAL_STRESS_WEIGHTS[comp]
    #             save_settings()
    #             st.rerun()
    # s = sum(st.session_state.STRESS_WEIGHTS.values())
    # for k in st.session_state.STRESS_WEIGHTS:
    #     st.session_state.STRESS_WEIGHTS[k] /= s

    st.subheader("📌 Expected Emotion Ranges")
    for emo in st.session_state.EXPECTED_RANGES.keys():
        col1, col2 = st.columns([4,1])
        with col1:
            low, high = st.session_state.EXPECTED_RANGES[emo]
            st.session_state.EXPECTED_RANGES[emo] = st.slider(
                f"{emo} % range", 0.0, 1.0, (low, high), 0.05, key=f"range_{emo}"
            )
        with col2:
            if st.button("Reset to Default ", key=f"reset_range_{emo}"):
                st.session_state.EXPECTED_RANGES[emo] = IDEAL_EXPECTED_RANGES[emo]
                save_settings()
                st.rerun()

    save_settings()
    st.success("✅ Admin settings saved. They will apply in Interview Dashboard.")
