import streamlit as st
import cv2
import math
import time
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import torch

# Set page configuration
st.set_page_config(page_title="Fake/Real Object Detection", layout="wide")

# Title and description
st.title("Fake vs Real Object Detection")
st.markdown("This application uses a YOLO model to detect whether objects are fake or real.")

# Sidebar configurations
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
source_option = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Video"])

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        # Force model to CPU if CUDA is causing issues
        model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main detection function
def process_frame(frame, model, confidence_threshold):
    class_names = ["fake", "real"]
    
    try:
        # Make a copy of the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Run inference with error handling
        results = model(frame, stream=False, verbose=False)  # Changed stream to False for stability
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                
                # Class Name
                cls = int(box.cls[0])
                
                # Only draw if confidence is above threshold
                if conf > confidence_threshold:
                    # Choose color based on class
                    if class_names[cls] == 'real':
                        color = (0, 255, 0)  # Green for real
                    else:
                        color = (0, 0, 255)  # Red for fake
                    
                    # Draw rectangle
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Add label with confidence
                    label = f'{class_names[cls].upper()} {int(conf*100)}%'
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Draw text background
                    cv2.rectangle(processed_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    
                    # Put text
                    cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return processed_frame
        
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        # Return original frame if processing fails
        return frame

# Function to handle webcam feed
def process_webcam(model):
    if model is None:
        st.error("Model failed to load. Cannot process webcam feed.")
        return
        
    st.header("Webcam Live Feed")
    
    # Create a placeholder for the video feed
    frame_window = st.empty()
    fps_display = st.empty()
    status_text = st.empty()
    
    # Button to start/stop the webcam
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    
    if start_button:
        status_text.text("Initializing webcam...")
        
        # Capture video from webcam
        cap = cv2.VideoCapture(0)  # Try default webcam
        
        if not cap.isOpened():
            # Try alternative camera index
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                status_text.error("Failed to open webcam. Please check your camera connection.")
                return
                
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
        
        status_text.text("Webcam started successfully. Press 'Stop Webcam' to end.")
        
        prev_time = 0
        processing_active = True
        
        while processing_active and cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Failed to capture frame from webcam")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            try:
                # Process the frame
                processed_frame = process_frame(frame, model, confidence_threshold)
                
                # Display FPS
                cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fps_display.text(f"FPS: {int(fps)}")
                
                # Convert from BGR to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Show the frame
                frame_window.image(processed_frame_rgb)
                
                # Check if stop button was clicked
                if stop_button:
                    processing_active = False
                    
            except Exception as e:
                status_text.error(f"Error during processing: {str(e)}")
                break
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Release resources
        cap.release()
        status_text.text("Webcam stopped.")
    
    elif stop_button:
        status_text.text("Webcam stopped.")

# Function to handle uploaded video
def process_uploaded_video(video_file, model):
    if model is None:
        st.error("Model failed to load. Cannot process video.")
        return
        
    st.header("Uploaded Video")
    
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    vf_path = tfile.name
    tfile.close()
    
    # Read the video file
    cap = cv2.VideoCapture(vf_path)
    
    if not cap.isOpened():
        st.error("Failed to open video file. The format might be unsupported.")
        os.unlink(vf_path)
        return
    
    # Create placeholder for video frames
    frame_window = st.empty()
    
    # Video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Progress bar
    progress_text = "Processing video..."
    progress_bar = st.progress(0)
    
    # Process each frame
    frame_index = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = process_frame(frame, model, confidence_threshold)
            
            # Convert from BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Show the frame
            frame_window.image(processed_frame_rgb)
            
            # Update progress
            frame_index += 1
            progress_bar.progress(min(frame_index / frame_count, 1.0))
            
            # Control playback speed (adjust as needed)
            time.sleep(1/(fps*2))  # Speed up playback
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    
    finally:
        # Release resources
        cap.release()
        if os.path.exists(vf_path):
            os.unlink(vf_path)
        progress_bar.empty()
        st.success("Video processing complete!")

# Main application
def main():
    # Specify model path
    model_path = st.text_input("Model Path", "C:\\Users\\JAYA SOORYA\\Downloads\\Real Vs Fake detection\\Model.pt")
    
    # Check if file exists
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at: {model_path}. Please check the path.")
        return
    
    # Load model button
    load_model_button = st.button("Load Model")
    
    model = None
    if load_model_button:
        with st.spinner("Loading model..."):
            try:
                model = load_model(model_path)
                if model:
                    st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return
    else:
        # Try to load model anyway for better UX
        try:
            model = load_model(model_path)
            if model:
                st.info("Model pre-loaded. Ready to use.")
        except:
            pass
    
    # Process based on selected source
    if source_option == "Webcam":
        process_webcam(model)
    else:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            process_uploaded_video(uploaded_video, model)

if __name__ == "__main__":
    main()