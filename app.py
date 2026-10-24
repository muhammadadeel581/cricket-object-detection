import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

# Page Configuration
st.set_page_config(page_title="Cricket Object Detection", layout="wide")

# App Header
st.title("🏏 Cricket Object Detection App")
st.write("Upload an image or video file to detect players, ball, umpire, and equipment.")

# Load Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar for Live Detection Info
st.sidebar.title("📊 Live Detection Info")
sidebar_placeholder = st.sidebar.empty()
sidebar_placeholder.markdown("Waiting for video/image to process...")

# File Uploader
uploaded_file = st.file_uploader("Choose a local image or video file...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    # Image Handling
    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Image Detection"):
            with st.spinner("Processing image..."):
                img_array = np.array(image)
                # conf=0.15 لگایا ہے تاکہ چھوٹی چیزیں (جیسے بال) بھی ڈیٹیکٹ ہوں
                results = model(img_array, conf=0.15)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Detection Result", use_column_width=True)
                
                # Show detected items list
                box_data = results[0].boxes
                class_names = results[0].names
                if box_data is not None and len(box_data) > 0:
                    detected_list = set([class_names[int(box.cls[0])] for box in box_data])
                    st.success(f"Detected Items: {', '.join(detected_list)}")
                else:
                    st.warning("No objects detected.")

    # Video Handling (Live Frame-by-Frame Processing with Sidebar Details)
    elif file_type == "mp4":
        st.video(uploaded_file)
        if st.button("Run Video Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            st.info("Processing video frames live with object details...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO prediction with lower confidence (conf=0.15) for better ball detection
                results = model(frame, conf=0.15)
                res_plotted = results[0].plot()
                
                # Extract detected class names for the current frame
                box_data = results[0].boxes
                class_names = results[0].names
                current_frame_objects = []
                if box_data is not None:
                    for box in box_data:
                        cls_id = int(box.cls[0])
                        obj_name = class_names[cls_id]
                        if obj_name not in current_frame_objects:
                            current_frame_objects.append(obj_name)
                
                # Update sidebar with currently detected items text
                if current_frame_objects:
                    sidebar_placeholder.markdown("**Current Frame Detected:**\n- " + "\n- ".join(current_frame_objects))
                else:
                    sidebar_placeholder.markdown("**Current Frame Detected:**\n- Searching...")
                
                # Convert OpenCV BGR to RGB for Streamlit display
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Show live frame
                st_frame.image(res_rgb, channels="RGB", use_column_width=True)
                
            cap.release()
            st.success("Video Detection Completed Successfully!")