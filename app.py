import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
from datetime import datetime

# ----------------- Setup -----------------
# Create folder for saving detection outputs
OUTPUT_DIR = "detected_elephants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load both models
models = {
    "Pretrained yolov8n.pt": YOLO("yolov8n.pt"),
    "Custom Trained Model (Dataset2)": YOLO("runs/detect/train4/weights/best.pt")
}

# Streamlit UI
st.title("🐘 Elephant Detection Dashboard")

# Sidebar - Choose model
model_choice = st.sidebar.selectbox("Choose detection model", list(models.keys()))
model = models[model_choice]

uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "avi", "mov", "jpg", "png"])

# ----------------- Processing -----------------
if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    detections = []  # store detection logs

    # If image
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        img = cv2.imread(tfile.name)
        results = model(img)
        annotated = results[0].plot()

        # Save annotated image
        save_path = os.path.join(OUTPUT_DIR, f"detected_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(save_path, annotated)

        st.image(annotated, channels="BGR", caption="Detection Result")

        # Collect detection info
        for box in results[0].boxes:
            cls = model.names[int(box.cls)]
            if cls.lower() == "elephant":
                detections.append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "detected_object": cls,
                    "confidence": float(box.conf),
                    "saved_file": save_path
                })

    # If video
    else:
        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.empty()

        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = os.path.join(OUTPUT_DIR, f"detected_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.avi")
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            if out is None:
                h, w, _ = annotated.shape
                out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))

            out.write(annotated)
            frame_window.image(annotated, channels="BGR")

            # Collect detection info
            for box in results[0].boxes:
                cls = model.names[int(box.cls)]
                if cls.lower() == "elephant":
                    detections.append({
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "detected_object": cls,
                        "confidence": float(box.conf),
                        "saved_file": out_path
                    })

        cap.release()
        if out:
            out.release()

    # ----------------- Display Detection Table -----------------
    if detections:
        df = pd.DataFrame(detections)
        st.success("✅ Elephants detected!")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No elephants detected.")
