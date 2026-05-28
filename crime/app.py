import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- 1. CONFIGURATION ---
MY_MODEL_FILE = 'crime_model.h5'
CLASSES = ["Safety/Normal", "Violence", "Crime", "Fist-Fight"]

# Page setup
st.set_page_config(page_title="Crime Detection AI", layout="centered")
st.title("🛡️ Crime & Violence Detection")
st.write("Upload an image to analyze for safety or criminal activity.")

# --- 2. LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    if not tf.io.gfile.exists(MY_MODEL_FILE):
        st.error(f"Model file '{MY_MODEL_FILE}' not found! Please place it in the same folder.")
        return None
    return tf.keras.models.load_model(MY_MODEL_FILE, compile=False)

model = load_my_model()

# --- 3. UI: FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Create two columns for UI
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Analysis Results")
        
        # 4. PREPROCESS
        # Get required size from model
        input_shape = model.input_shape
        req_h, req_w = input_shape[1], input_shape[2]
        
        # Resize and normalize
        img = cv2.resize(opencv_image, (req_w, req_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # 5. INFERENCE
        preds = model.predict(img)
        class_idx = np.argmax(preds[0])
        label = CLASSES[class_idx]
        confidence = preds[0][class_idx]

        # 6. DISPLAY RESULTS
        if label in ["Crime", "Fist-Fight"]:
            st.error(f"**ALERT: {label} Detected!**")
        else:
            st.success(f"**Status: {label}**")
            
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

        # Show probability bar chart
        st.write("#### Class Probabilities:")
        chart_data = {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}
        st.bar_chart(chart_data)

# --- INSTRUCTIONS ---
# Save this file as streamlit_app.py
# Run it using: streamlit run streamlit_app.py