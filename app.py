
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_cnn.keras"
    )

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Upload an image of a handwritten digit and "
    "the CNN model will predict the digit."
)

st.divider()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(
        image,
        width=200
    )

    # Convert to numpy
    image_array = np.array(image)

    # Resize
    image_resized = Image.fromarray(
        image_array
    ).resize((28, 28))

    image_array = np.array(
        image_resized
    ).astype("float32")

    # Automatically handle white-background images
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Normalize
    image_array = image_array / 255.0

    # Add dimensions for CNN
    image_input = image_array.reshape(
        1, 28, 28, 1
    )

    # Prediction
    prediction = model.predict(
        image_input,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction) * 100
    )

    # -----------------------------
    # Display result
    # -----------------------------
    st.divider()

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Digit",
            predicted_digit
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # -----------------------------
    # Probability chart
    # -----------------------------
    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots()

    ax.bar(
        range(10),
        prediction
    )

    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(10))

    st.pyplot(fig)
