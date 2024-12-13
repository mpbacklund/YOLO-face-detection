import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from flask import Flask, request, send_file
import io
from PIL import Image
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure an image file is included in the request
    if "image" not in request.files:
        return {"error": "No image provided"}, 400

    # Load the image file from the request
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run the YOLO model
    results = model(image)

    # Process each detection
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)  # Get box coordinates
            width, height = x2 - x1, y2 - y1  # Calculate box dimensions
            image = overlay_image(image, cls, x1, y1, width, height)

    # Convert the processed image to bytes
    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # Return the processed image
    return send_file(img_bytes, mimetype="image/jpeg")


# Function to overlay an image on top of another at a specific location
def overlay_image(background, overlay_type, x, y, width, height):
    if overlay_type == 0 or overlay_type == 1:
        overlay_path = "assets/eye2.png"
    elif overlay_type == 2:
        overlay_path = "assets/nose.png"
    else:
        overlay_path = "assets/mouth.png"

    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    resized_overlay = cv2.resize(overlay, (width, height) if overlay_type == 3 else (width, width))
    edited_image = cvzone.overlayPNG(background, resized_overlay, [x, y])

    return edited_image


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
