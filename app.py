import gradio as gr
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from PIL import Image

# 🔥 Rebuild SAME architecture used during training
base_model = MobileNetV2(
    weights=None,              # ⚠️ IMPORTANT: No imagenet download
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

# ✅ Load your trained weights
model.load_weights("skin_cancer_model.weights.h5")

# Class labels
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    result = class_labels[class_index]

    return f"Prediction: {result.upper()} | Confidence: {confidence:.2f}%"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Skin Cancer Detection",
    description="Upload a skin image to detect cancer type."
)

interface.launch()