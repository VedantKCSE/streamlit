from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from PIL import Image
import io
from scipy.stats import skew, kurtosis
import mahotas
from skimage.feature import hog
from skimage import exposure
import base64
from io import BytesIO

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load("svm_model.pkl")

# Helper Functions
def preprocess_image(image_bytes):
    """
    Preprocess the image for prediction:
    1. Convert to grayscale.
    2. Resize to 150x150.
    3. Apply Gaussian Blur and CLAHE.
    """
    # Load image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = np.array(image)

    # Resize to 150x150
    image = cv2.resize(image, (150, 150))

    # Apply Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    return image

def extract_statistical_features(image):
    flat_image = image.flatten()
    mean_val = np.mean(flat_image)
    variance_val = np.var(flat_image)
    skewness_val = skew(flat_image)
    kurtosis_val = kurtosis(flat_image)
    return [mean_val, variance_val, skewness_val, kurtosis_val]

def extract_texture_features(image):
    texture_features = mahotas.features.haralick(image)
    return np.mean(texture_features, axis=0)

def extract_edge_features(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edge_sum = np.sum(gradient_magnitude)
    edge_std = np.std(gradient_magnitude)
    return [edge_sum, edge_std]

def extract_hog_features(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

def extract_features(image):
    statistical = extract_statistical_features(image)
    texture = extract_texture_features(image)
    edges = extract_edge_features(image)
    hog_features = extract_hog_features(image)
    combined_features = np.concatenate([statistical, texture, edges, hog_features])
    return combined_features

def encode_image_to_base64(image):
    pil_image = Image.fromarray(image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file:
        return jsonify({"error": "File is empty"}), 400

    try:
        # Preprocess the image
        image = preprocess_image(file.read())

        # Extract features
        features = extract_features(image)

        # Encode preprocessed image for display in Streamlit
        preprocessed_image_b64 = encode_image_to_base64(image)

        # Predict using the model
        prediction = model.predict([features])
        result = "Pneumonia" if prediction[0] == 1 else "Normal"

        # Return the prediction and preprocessed image
        return jsonify({
            "prediction": result,
            "preprocessed_image": preprocessed_image_b64,
            "features": features.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
