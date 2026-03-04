from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image

app = Flask(__name__)

# 🔥 Rebuild Architecture
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

# ✅ Load weights
model.load_weights("skin_weights.weights.h5")

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            class_index = np.argmax(preds)

            prediction = class_labels[class_index].upper()
            confidence = round(float(np.max(preds) * 100), 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)