# Pneumonia Detection Project

This project aims to detect pneumonia in chest X-ray images using various machine learning methods. The dataset used for this project is the **Kaggle Chest X-ray Pneumonia Dataset**: [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Methods Used

Three different methods are employed for pneumonia detection, and the code for each method is available on the corresponding branches. Below are the methods used:

1. **Feature Extraction and SVM Classifier**:
    - The model uses feature extraction techniques from chest X-ray images to classify whether the image shows signs of pneumonia. The features are then fed into an **SVM (Support Vector Machine)** model for classification.
    - The code for this method is available in the `feature-extraction-svm` branch.

2. **Convolutional Neural Networks (CNN)**:
    - A CNN-based model is implemented for image classification to automatically detect pneumonia from X-ray images. CNNs are effective for image-related tasks and provide high accuracy in detecting diseases from medical images.
    - The code for this method is available in the `cnn-method` branch.

3. **Transfer Learning with Pre-trained Models**:
    - Transfer learning is used by leveraging pre-trained models such as VGG16 or ResNet to classify pneumonia in chest X-ray images. The model is fine-tuned to suit the specific dataset of chest X-rays.
    - The code for this method is available in the `transfer-learning` branch.

## Project Information

- **Dataset**: The project utilizes the **Chest X-ray Pneumonia Dataset** available on Kaggle, which consists of labeled X-ray images categorized as pneumonia or normal.
- **Model Evaluation**: The models are evaluated using standard metrics such as accuracy, precision, recall, and F1-score. Cross-validation is used to ensure the reliability of the model's predictions.


