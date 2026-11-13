import webbrowser
from threading import Timer
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model("skin_disease_model.keras")

classes = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":
        file = request.files["image"]

        if file:
           upload_folder = os.path.join(
              app.root_path,
              "static",
              "uploads"
    )

        if not os.path.exists(upload_folder):
          os.makedirs(upload_folder)

        path = os.path.join(upload_folder, file.filename)
        file.save(path)

        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        prediction = classes[np.argmax(pred)]

    return render_template("index.html",
                           prediction=prediction)
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)