import os
import json
import cv2
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# ==============================
# Flask App
# ==============================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# Load Model
# ==============================

MODEL_PATH = os.path.join(BASE_DIR, "Butterfly_Final.keras")

model = load_model(MODEL_PATH)

# ==============================
# Load Label Map
# ==============================

LABEL_PATH = os.path.join(BASE_DIR, "label_map.json")

with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)

# Reverse Label
reverse_label = {v: k for k, v in label_map.items()}

# ==============================
# Image Preprocessing
# ==============================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ==============================
# Home Page
# ==============================

@app.route("/")
def home():

    return render_template("index.html")

# ==============================
# Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image"

    filename = file.filename

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    image = preprocess_image(filepath)

    prediction = model.predict(image, verbose=0)

    class_index = int(np.argmax(prediction))

    confidence = round(float(np.max(prediction) * 100), 2)

    species = reverse_label[class_index]

    return render_template(
        "result.html",
        image=filename,
        species=species,
        confidence=confidence
    )

# ==============================
# Run
# ==============================

if __name__ == "__main__":

    app.run(debug=True)