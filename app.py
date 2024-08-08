import os
import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow import keras
import cv2  # For video capture
from dotenv import load_dotenv

load_dotenv()

# Load the trained model
model = keras.models.load_model("emotion_detection_model.h5")

# Load class indices from JSON file
class_indices = json.load(open("class.json"))

# Function to preprocess image
def preprocess_image(image):
    image_rgb = image.convert("RGB")
    resized_image = image_rgb.resize((224, 224))
    normalized_image = np.array(resized_image, dtype=np.float32) / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to predict emotion
def predict_emotion(model, image, class_indices):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit interface
st.title("Live Face Emotion Recognition")
st.subheader("Made by Dhiraj")
st.text("Lets Check")

# Video capture from webcam
cap = cv2.VideoCapture(0)

# Placeholder for the video feed
frame_placeholder = st.empty()

# Loop to continuously capture frames and predict emotions
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

    # Convert the frame to RGB format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Predict the emotion
    emotion = predict_emotion(model, image, class_indices)

    # Display the emotion on the frame
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion in Streamlit
    frame_placeholder.image(frame, channels="BGR")

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
