import streamlit as st
import cv2
import numpy as np
import joblib
import skimage
from skimage.feature import hog
from PIL import Image
import mahotas



# Load the trained Random Forest model
rf_model = joblib.load('rf_model.pkl')  # Assuming the model is saved as 'rf_model.pkl'

# Function to extract HOG features from an image
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

# Function to preprocess the image and extract features
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Resize image to match the size expected by the model
    gray_image_resized = cv2.resize(gray_image, (128, 128))  # You can change this based on your input size
    
    # Extract features
    features = extract_hog_features(gray_image_resized)
    
    return features

# Streamlit UI
st.title('Pneumonia Detection from X-Ray Image')

# File uploader widget for image input
uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and extract features
    image = Image.open(uploaded_file)
    features = preprocess_image(image)
    
    # Reshape the features to match the model input shape
    features = features.reshape(1, -1)
    
    # Predict using the Random Forest model
    prediction = rf_model.predict(features)
    
    # Display the prediction
    if prediction == 0:
        st.write("Prediction: **Normal**")
    else:
        st.write("Prediction: **Pneumonia**")