import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Cricket Object Detection App", layout="centered")

# App Header
st.title("🏏 Cricket Object Detection App")
st.write("Upload an image or video file to detect players, ball, umpires, and wickets.")

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
uploaded_file = st.file_uploader("Choose a local image or video file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # Image Handling
    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        
        if st.button("Run Image Detection"):
            with st.spinner("Processing image..."):
                img_array = np.array(image)
                results = model(img_array, conf=0.03)
                res_plotted = results[0].plot()
                
                st.image(res_plotted, caption="Detection Result", channels="RGB", width=400)
                
                box_data = results[0].boxes
                class_names = results[0].names
                if box_data is not None and len(box_data) > 0:
                    detected_list = set([class_names[int(box.cls[0])] for box in box_data])
                    sidebar_placeholder.markdown(f"**Detected Items:**\n- " + "\n- ".join(detected_list))
                    st.success(f"Detected Items: {', '.join(detected_list)}")
                else:
                    sidebar_placeholder.markdown("**Detected Items:**\n- No objects detected.")
                    st.warning("No objects detected.")

    # Video Handling (Live Frame-by-Frame Processing with Sidebar Info)
    elif file_type == "mp4":
        st.video(uploaded_file, format="video/mp4")
        
        if st.button("Run Video Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            st.info("Processing video frames live with object detection...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame, conf=0.03)
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
                
                # Update sidebar with currently detected items in real-time
                if current_frame_objects:
                    sidebar_placeholder.markdown("**Current Frame Detected:**\n- " + "\n- ".join(current_frame_objects))
                else:
                    sidebar_placeholder.markdown("**Current Frame Detected:**\n- No objects detected")
                
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, channels="RGB", width=400)
            
            cap.release()
            sidebar_placeholder.markdown("**Status:** Completed")
            st.success("Video Detection Completed Successfully!")
