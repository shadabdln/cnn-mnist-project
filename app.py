import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="CNN Digit Classifier", layout="centered")

# Title
st.title("🧠 Handwritten Digit Classifier")
st.write("Upload an image of a digit (0–9) and get prediction")

# Load Model
try:
    model = tf.keras.models.load_model("cnn_mnist_model.h5")
    st.success("Model Loaded Successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# File Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Show Image
    with col1:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        st.image(image, caption="Uploaded Image", width=150)

    # Prediction
    with col2:
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        result = np.argmax(prediction)

        st.success(f"✅ Predicted Digit: {result}")

        st.write("### 🔢 Prediction Confidence:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{i}: {prob:.4f}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using CNN & Streamlit")