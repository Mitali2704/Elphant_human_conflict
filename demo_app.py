"""
demo_app.py

Streamlit app that runs YOLO detection (image/video), plays alert sound, sends JSON events
to a Flask backend, saves detected images/videos and events to CSV, and shows detections in a table + map.
"""

import os
import tempfile
import time
import threading
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import requests

# ---------------------------
# Create folders for saving
# ---------------------------
BASE_DIR = "detected_elephants"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
EVENTS_FILE = os.path.join(BASE_DIR, "elephant_events.csv")

# ---------------------------
# Alert sound helper
# ---------------------------
def _make_play_alert():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        def play_alert(sound_file):
            try:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play(-1)  # loop
            except Exception as e:
                st.warning(f"Unable to play sound via pygame: {e}")
        return play_alert

    except Exception:
        try:
            from playsound import playsound
            def play_alert(sound_file):
                threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
            return play_alert
        except Exception:
            def play_alert(sound_file):
                st.warning("⚠️ No audio player available (pygame/playsound missing).")
            return play_alert

play_alert = _make_play_alert()

def stop_alert():
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass

# ---------------------------
# Location helper
# ---------------------------
def get_device_location(fallback=(20.315, 85.825)):
    try:
        r = requests.get("https://ipinfo.io", timeout=3)
        if r.ok:
            loc = r.json().get("loc")
            if loc:
                lat, lon = map(float, loc.split(","))
                return lat, lon
    except Exception:
        pass
    return fallback

# ---------------------------
# YOLO model loading
# ---------------------------
try:
    from ultralytics import YOLO
except Exception:
    st.error("⚠️ ultralytics is not installed. Run: `pip install ultralytics`")
    raise

MODEL_FILENAME = "yolov8n.pt"
model_to_load = MODEL_FILENAME if os.path.exists(MODEL_FILENAME) else MODEL_FILENAME

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_to_load)
st.sidebar.success(f"✅ Loaded model: {model_to_load}")

# ---------------------------
# UI Controls
# ---------------------------
st.title("🐘 Elephant Detection Dashboard")

st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detection confidence threshold", 0.1, 0.99, 0.25, 0.01)
alert_threshold = st.sidebar.slider("Alert trigger confidence", 0.5, 0.99, 0.8, 0.01)
backend_url = st.sidebar.text_input("Backend event URL", value="http://127.0.0.1:5000/upload_event")
sound_file = st.sidebar.text_input("Alert sound file", value="aleart_sound.wav")
frame_skip = st.sidebar.slider("Frame skip (video)", 1, 30, 5)

if st.sidebar.button("⏹️ Stop Alarm"):
    stop_alert()
    st.sidebar.info("Alarm stopped.")

if "events" not in st.session_state:
    st.session_state.events = []
if "last_event_time" not in st.session_state:
    st.session_state.last_event_time = 0.0

THROTTLE_SECONDS = 20

# ---------------------------
# Helpers
# ---------------------------
def extract_detections_from_result(results_obj):
    annotated = results_obj.plot()
    cls_list, conf_list = [], []
    try:
        cls_list = [int(x) for x in results_obj.boxes.cls.tolist()]
        conf_list = [float(x) for x in results_obj.boxes.conf.tolist()]
    except Exception:
        for box in results_obj.boxes:
            try:
                cls_list.append(int(box.cls))
                conf_list.append(float(box.conf))
            except:
                pass
    class_names = [results_obj.names[cid] for cid in cls_list]
    return class_names, conf_list, annotated

def send_event_to_backend(event_dict):
    try:
        resp = requests.post(backend_url, json=event_dict, timeout=4)
        return resp.status_code, resp.text
    except Exception as e:
        return None, str(e)

def handle_elephant_detection(max_conf, herd_size, detected_img=None, detected_video=None):
    now_ts = time.time()
    if now_ts - st.session_state.last_event_time < THROTTLE_SECONDS:
        st.info("Event throttled (recently sent).")
        return

    if os.path.exists(sound_file):
        play_alert(sound_file)
    else:
        st.warning(f"⚠️ Sound file '{sound_file}' not found!")

    lat, lon = get_device_location()

    # Save detected image
    image_path = None
    if detected_img is not None:
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMAGES_DIR, f"elephant_{timestamp_str}.jpg")
        if isinstance(detected_img, np.ndarray):
            cv2.imwrite(image_path, detected_img)
        else:
            detected_img.save(image_path)

    # Save detected video clip if any
    video_path = None
    if detected_video is not None:
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(VIDEOS_DIR, f"elephant_{timestamp_str}.mp4")
        cv2.VideoWriter(detected_video, video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (detected_video.shape[1], detected_video.shape[0]))

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device_id": "SIM-01",
        "latitude": lat,
        "longitude": lon,
        "detected_object": "elephant",
        "confidence": float(max_conf),
        "herd_size": int(herd_size),
        "image_path": image_path,
        "video_path": video_path
    }

    # Save event to CSV
    st.session_state.events.append(event)
    st.session_state.last_event_time = now_ts
    try:
        if os.path.exists(EVENTS_FILE):
            df = pd.read_csv(EVENTS_FILE)
            df = pd.concat([df, pd.DataFrame([event])], ignore_index=True)
        else:
            df = pd.DataFrame([event])
        df.to_csv(EVENTS_FILE, index=False)
    except Exception as e:
        st.error(f"Failed to save event to CSV: {e}")

    # Send to backend
    def _send_thread(ev):
        status, msg = send_event_to_backend(ev)
        if status == 200:
            st.success("📡 Event sent successfully.")
        else:
            st.error(f"Backend error (status={status}): {msg}")
    threading.Thread(target=_send_thread, args=(event,), daemon=True).start()

# ---------------------------
# Upload handler
# ---------------------------
uploaded = st.file_uploader("Upload image/video", type=["jpg","jpeg","png","mp4","avi","mov"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp_path = os.path.join(tempfile.gettempdir(), f"tmp{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    if uploaded.type.startswith("image"):
        img = Image.open(tmp_path).convert("RGB")
        img_array = np.array(img)[:, :, ::-1]

        results = model.predict(img_array, conf=conf_threshold, imgsz=640)
        result0 = results[0]
        class_names, confs, annotated = extract_detections_from_result(result0)

        st.image(annotated, channels="BGR", caption="Detection result", use_column_width=True)

        elephant_confs = [c for cls, c in zip(class_names, confs) if cls.lower()=="elephant" and c>=alert_threshold]
        if elephant_confs:
            max_conf, herd_size = max(elephant_confs), len(elephant_confs)
            st.success(f"🚨 Elephant detected! Herd size={herd_size}, Max conf={max_conf:.2f}")
            handle_elephant_detection(max_conf, herd_size, detected_img=img)
        else:
            st.info("✅ No elephant detected above alert threshold.")

    else:  # Video
        st.info("Processing video...")
        cap = cv2.VideoCapture(tmp_path)
        frame_window = st.empty()
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx +=1
            if frame_idx % frame_skip != 0:
                continue

            results = model.predict(frame, conf=conf_threshold, imgsz=640)
            res0 = results[0]
            cls_names, confs, annotated = extract_detections_from_result(res0)
            frame_window.image(annotated, channels="BGR")

            elephant_confs = [c for cls, c in zip(cls_names, confs) if cls.lower()=="elephant" and c>=alert_threshold]
            if elephant_confs:
                max_conf, herd_size = max(elephant_confs), len(elephant_confs)
                handle_elephant_detection(max_conf, herd_size, detected_img=annotated)

        cap.release()

# ---------------------------
# Show detection events
# ---------------------------
st.markdown("---")
st.header("📋 Detection Events")

if st.session_state.events:
    df_events = pd.DataFrame(st.session_state.events)
    st.dataframe(df_events.sort_values(by="timestamp", ascending=False), use_container_width=True)
    try:
        map_df = df_events[["latitude","longitude"]].dropna().astype(float)
        if not map_df.empty:
            st.map(map_df)
    except:
        pass
else:
    st.info("No detection events yet.")
