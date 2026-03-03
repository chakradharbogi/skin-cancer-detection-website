from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from PIL import Image
import os

app = Flask(__name__)

# ✅ Load FULL model (No rebuilding)
model = keras.models.load_model("skin_cancer_model.keras")

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

    # Preprocess image
    img = Image.open(filepath).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)