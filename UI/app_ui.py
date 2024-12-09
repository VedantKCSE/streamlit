import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Send the file to the backend
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")

            # Display the preprocessed image
            preprocessed_image_b64 = result['preprocessed_image']
            preprocessed_image_bytes = BytesIO(base64.b64decode(preprocessed_image_b64))
            preprocessed_image = Image.open(preprocessed_image_bytes)
            st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

            # Display the extracted features
            # st.write("Extracted Features:")
            # st.json(result['features'])
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
