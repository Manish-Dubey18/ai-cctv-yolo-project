import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))

from fire_detection import FireDetector
from crime_classifier import CrimeClassifier
from anomaly_detection import AnomalyDetector
from crowd_detection import CrowdDetector

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Public Safety Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .alert-box {
        padding: 10px 16px;
        border-radius: 8px;
        margin: 6px 0;
        font-weight: bold;
        font-size: 15px;
    }
    .alert-fire   { background: #ff4b4b22; border-left: 4px solid #ff4b4b; color: #ff4b4b; }
    .alert-crime  { background: #ffa50022; border-left: 4px solid #ffa500; color: #ffa500; }
    .alert-anomaly{ background: #a855f722; border-left: 4px solid #a855f7; color: #a855f7; }
    .alert-crowd  { background: #3b82f622; border-left: 4px solid #3b82f6; color: #3b82f6; }
    .alert-normal { background: #22c55e22; border-left: 4px solid #22c55e; color: #22c55e; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for key, val in {
    "fire_count": 0, "crime_count": 0,
    "anomaly_count": 0, "crowd_count": 0,
    "alert_log": [], "running": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "fire": FireDetector(model_path="models/fire_best.pt", conf=0.5),
        "crime": CrimeClassifier(model_path="models/crime_model.h5"),
        "anomaly": AnomalyDetector(model_path="models/anomaly_model.h5"),
        "crowd": CrowdDetector(model_path="models/crowd_model.h5", crowd_threshold=10)
    }

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🛡️ Public Safety Monitoring Dashboard")
st.markdown("Real-time detection of **Fire · Crime · Anomaly · Crowd**")
st.divider()

# ─────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    source = st.radio("📹 Input Source", ["Webcam", "Video File"])

    video_path = None
    if source == "Video File":
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            video_path = tfile.name

    st.divider()
    st.subheader("🔧 Detection Settings")
    fire_conf = st.slider("Fire Confidence", 0.1, 1.0, 0.5, 0.05)
    anomaly_thresh = st.slider("Anomaly Threshold", 0.1, 1.0, 0.6, 0.05)
    crowd_thresh = st.slider("Crowd Alert (people)", 5, 50, 10, 1)

    st.divider()
    st.subheader("🎯 Active Detectors")
    use_fire    = st.toggle("🔥 Fire Detection",    value=True)
    use_crime   = st.toggle("🚨 Crime Detection",   value=True)
    use_anomaly = st.toggle("⚠️ Anomaly Detection", value=True)
    use_crowd   = st.toggle("👥 Crowd Detection",   value=True)

    st.divider()
    start = st.button("▶️ Start", use_container_width=True, type="primary")
    stop  = st.button("⏹️ Stop",  use_container_width=True)

    if start: st.session_state.running = True
    if stop:  st.session_state.running = False

    if st.button("🔄 Reset Counters", use_container_width=True):
        st.session_state.fire_count = 0
        st.session_state.crime_count = 0
        st.session_state.anomaly_count = 0
        st.session_state.crowd_count = 0
        st.session_state.alert_log = []

# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────
col_vid, col_info = st.columns([2, 1])

with col_vid:
    st.subheader("📷 Live Feed")
    frame_window = st.empty()
    status_bar = st.empty()

with col_info:
    st.subheader("📊 Alert Counters")
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    fire_metric    = m1.metric("🔥 Fire",    st.session_state.fire_count)
    crime_metric   = m2.metric("🚨 Crime",   st.session_state.crime_count)
    anomaly_metric = m3.metric("⚠️ Anomaly", st.session_state.anomaly_count)
    crowd_metric   = m4.metric("👥 Crowd",   st.session_state.crowd_count)

    st.divider()
    st.subheader("🗂️ Alert Log")
    log_container = st.container(height=300)

# ─────────────────────────────────────────
# ALERT HELPER
# ─────────────────────────────────────────
def log_alert(kind, message):
    timestamp = time.strftime("%H:%M:%S")
    entry = {"time": timestamp, "kind": kind, "msg": message}
    st.session_state.alert_log.insert(0, entry)
    st.session_state.alert_log = st.session_state.alert_log[:50]  # keep last 50

def render_log():
    with log_container:
        if not st.session_state.alert_log:
            st.info("No alerts yet.")
        for entry in st.session_state.alert_log[:15]:
            css = {
                "fire": "alert-fire", "crime": "alert-crime",
                "anomaly": "alert-anomaly", "crowd": "alert-crowd"
            }.get(entry["kind"], "alert-normal")
            st.markdown(
                f'<div class="alert-box {css}">[{entry["time"]}] {entry["msg"]}</div>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────
# DETECTION LOOP
# ─────────────────────────────────────────
if st.session_state.running:
    models = load_models()

    # Update model settings from sliders
    models["fire"].conf = fire_conf
    models["anomaly"].threshold = anomaly_thresh
    models["crowd"].crowd_threshold = crowd_thresh

    # Open video source
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if video_path is None:
            st.warning("⚠️ Please upload a video file first.")
            st.stop()
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open video source.")
        st.stop()

    frame_count = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            if source == "Video File":
                status_bar.info("✅ Video ended.")
            break

        frame_count += 1
        display_frame = frame.copy()
        alerts_this_frame = []

        # ── Fire Detection ──
        if use_fire:
            fire_frame, fire_det, fire_cnt = models["fire"].detect(frame)
            if fire_det:
                st.session_state.fire_count += 1
                alerts_this_frame.append(("fire", f"🔥 Fire detected! ({fire_cnt} instance(s))"))
                display_frame = fire_frame

        # ── Crime Classification ──
        if use_crime:
            crime_label, crime_conf_val, crime_det = models["crime"].detect(frame)
            if crime_det:
                st.session_state.crime_count += 1
                alerts_this_frame.append(("crime", f"🚨 Crime: {crime_label} ({crime_conf_val:.0%})"))
            cv2.putText(display_frame, f"Crime: {crime_label} {crime_conf_val:.0%}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # ── Anomaly Detection ──
        if use_anomaly:
            anom_label, anom_score, anom_det = models["anomaly"].detect(frame)
            if anom_det:
                st.session_state.anomaly_count += 1
                alerts_this_frame.append(("anomaly", f"⚠️ Anomaly detected! Score: {anom_score:.2f}"))
            cv2.putText(display_frame, f"Anomaly: {anom_label} {anom_score:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 255), 2)

        # ── Crowd Detection ──
        if use_crowd:
            crowd_frame, people_cnt, crowd_alert = models["crowd"].detect(frame)
            if crowd_alert:
                st.session_state.crowd_count += 1
                alerts_this_frame.append(("crowd", f"👥 Crowd alert! {people_cnt} people detected"))
            display_frame = crowd_frame

        # ── Log alerts ──
        for kind, msg in alerts_this_frame:
            log_alert(kind, msg)

        # ── Show frame ──
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(rgb, channels="RGB", use_container_width=True)
        status_bar.caption(f"Frame: {frame_count} | "
                           f"🔥 {st.session_state.fire_count} | "
                           f"🚨 {st.session_state.crime_count} | "
                           f"⚠️ {st.session_state.anomaly_count} | "
                           f"👥 {st.session_state.crowd_count}")

        render_log()

    cap.release()
    st.session_state.running = False

else:
    # Placeholder when not running
    frame_window.image(
        np.zeros((480, 640, 3), dtype=np.uint8),
        channels="RGB", use_container_width=True
    )
    status_bar.caption("Press ▶️ Start to begin monitoring.")
    render_log()