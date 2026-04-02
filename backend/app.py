import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

from security.aes_encryption import encrypt_image
from explainability.gradcam import generate_gradcam

app = Flask(__name__)

model = tf.keras.models.load_model("model/saved_model.h5")

users = {
    "admin":"password123"
}

def preprocess(img):

    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    return img


@app.route("/login", methods=["POST"])
def login():

    data = request.json

    username = data["username"]
    password = data["password"]

    if username in users and users[username]==password:
        return jsonify({"message":"Login Successful"})

    return jsonify({"message":"Invalid credentials"}),401


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    encrypted = encrypt_image(img.tobytes())

    img_array = preprocess(img)

    prediction = model.predict(img_array)[0][0]
    print("Prediction score:", prediction)

    result = "Diseased" if prediction > 0.7 else "Healthy"

    heatmap = generate_gradcam(model, img_array)

    return jsonify({
        "prediction": result,
        "confidence": float(prediction)
    })

@app.route("/")
def home():
    return """
    <h2>Medical Image Diagnosis</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Predict">
    </form>
    """


if __name__ == "__main__":
    app.run(debug=True)