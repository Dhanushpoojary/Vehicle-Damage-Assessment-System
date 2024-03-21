import argparse
import io
import os
import tempfile
from PIL import Image

import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect

app = Flask(__name__, static_folder='static')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()


# Price dictionary for different damage labels
price_dict = {
    'Scratch': 750,
    'Deformation': 1500,
    'Broken Glass': 5000,
    'Broken': 40000
}


def process_image(img):
    img2 = img.resize((224, 224))  # Resize the PIL image object
    x = tf.keras.utils.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    return x


def convert_image_to_jpeg(img_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img_rgb = img.convert('RGB')
            img_rgb.save(tmp_file, format='JPEG')
        return tmp_file.name


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()

        # Convert image to JPEG format
        img_path = convert_image_to_jpeg(img_bytes)
        img = Image.open(img_path)

        # Object detection with YOLOv5
        results = model(img)  # inference
        results.render()  # updates results.ims with boxes and labels
        Image.fromarray(results.ims[0]).save("static/images/image0.jpg")

        # Get detected objects, confidence scores, and labels
        detected_objects = results.names
        confidence_scores = results.xyxy[0][:, 4].cpu().numpy()
        labels = results.xyxy[0][:, 5].cpu().numpy()

        # Pre-process detected objects and confidence scores
        detected_objects_with_scores = [
            (obj, confidence, detected_objects[int(label)])
            for obj, confidence, label in zip(detected_objects, confidence_scores, labels)
        ]

        # Calculate total price for detected labels
        total_price = 0
        detected_objects_with_prices = []

        for obj, confidence, label in detected_objects_with_scores:
            if label in price_dict:
                price = price_dict[label]
                # Multiply the price by the confidence score and round off to 2 decimals
                price_with_confidence = round(price * confidence, 2)
                total_price += price_with_confidence
                detected_objects_with_prices.append((obj, confidence, label, price_with_confidence))


        # Merge images
        image1 = Image.open('static/images/image0.jpg')
        image2 = Image.open('static/images/screenshot.jpg')
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB', (3 * image2_size[0], image2_size[1]), (250, 250, 250))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1_size[0], 0))
        new_image.save("static/images/merged_image.jpg", "JPEG")


        os.remove(img_path)  # Remove temporary image file

        return render_template("display_image.html",
                               image_path="static/images/image0.jpg",
                               detected_objects_with_scores=detected_objects_with_prices,
                               total_price=round(total_price))

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)