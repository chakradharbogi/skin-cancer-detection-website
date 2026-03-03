import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

st.title("Skin Cancer Detection App")

# Build model
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Load weights
model.load_weights("skin_weights.weights.h5")

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    st.success(f"Prediction: {class_labels[class_index]}")
    st.info(f"Confidence: {confidence:.2f}%")