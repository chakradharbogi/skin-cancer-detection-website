from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

app = Flask(__name__)

# 🔥 Build Model Architecture (Same as Training)

base_model = MobileNetV2(
    weights='imagenet',
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

# ✅ Load trained weights
model.load_weights("skin_cancer_model.weights.h5")

# HAM10000 Classes
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction_text="No file selected")

    upload_folder = "static"
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Image Preprocessing
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    result = class_labels[class_index]

    return render_template(
        'index.html',
        prediction_text=f"Prediction: {result.upper()}",
        confidence=f"Confidence: {confidence:.2f}%",
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)