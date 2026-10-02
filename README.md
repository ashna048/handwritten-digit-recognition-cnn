
# Handwritten Digit Recognition using CNN

## Project Overview

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from 0 to 9.

The model is trained using the MNIST handwritten digit dataset and is capable of classifying previously unseen handwritten digit images.

A Streamlit web application is also included to provide an interactive prediction interface.

---

## Features

- Handwritten digit classification from 0 to 9
- Image preprocessing and normalization
- CNN-based feature extraction
- Multiclass classification using Softmax
- Model evaluation using accuracy, precision, recall and F1-score
- Confusion matrix visualization
- Prediction confidence scores
- Interactive Streamlit application

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pillow

---

## Dataset

The project uses the MNIST handwritten digit dataset.

Dataset characteristics:

- 60,000 training images
- 10,000 testing images
- 10 digit classes (0-9)
- Image size: 28 × 28 pixels
- Grayscale images

---

## Data Preprocessing

The following preprocessing steps were applied:

1. Converted image pixel values from the range 0-255 to 0-1.
2. Added a channel dimension for CNN processing.
3. Used the training dataset for model learning and validation.
4. Evaluated the final model on the separate MNIST test dataset.

---

## CNN Architecture

The model consists of:

1. Input layer
2. Convolutional layer with 32 filters
3. Max pooling layer
4. Convolutional layer with 64 filters
5. Max pooling layer
6. Flatten layer
7. Dense layer with 128 neurons
8. Dropout layer
9. Output layer with 10 neurons using Softmax activation

---

## Model Performance

The trained CNN achieved:

**Test Accuracy: 99.20%**

The model was evaluated on 10,000 unseen MNIST test images.

---

## Interactive Application

A Streamlit application allows users to upload a handwritten digit image.

The application:

1. Accepts an image upload.
2. Converts the image to grayscale.
3. Resizes the image to 28 × 28 pixels.
4. Normalizes the image.
5. Sends the image to the trained CNN.
6. Displays the predicted digit.
7. Displays the prediction confidence.
8. Displays probabilities for all ten digits.

---
## Results

The CNN achieved **99.20% test accuracy** on 10,000 unseen MNIST test images.

The project also includes an interactive Streamlit application where users can upload a handwritten digit image and receive:

- Predicted digit
- Prediction confidence
- Probability distribution across all 10 classes

----

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ashna048-handwritten-digit-recognition-cnn-app-rklhxy.streamlit.app/)

👉 **[Open the Live Application](https://ashna048-handwritten-digit-recognition-cnn-app-rklhxy.streamlit.app/)**

-----

## Project Structure

```text
Handwritten-Digit-Recognition/
│
├── Handwritten_Digit_Recognition_CNN.ipynb
├── handwritten_digit_cnn.keras
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
