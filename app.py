from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# model load
model = tf.keras.models.load_model("model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("L")
    img = img.resize((28,28))
    img = np.array(img) / 255.0
    img = img.reshape(1,28,28,1)

    pred = model.predict(img)
    result = np.argmax(pred)

    return str(result)

if __name__ == "__main__":
    app.run(debug=True)
