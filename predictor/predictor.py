import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from flask import Flask, send_file

app = Flask(__name__)

# Load your trained model
model = YOLO("best.pt")

@app.route("/predict")
def predict(image):
    # Load the test image
    image = cv2.imread(image)

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

    return send_file(image, mimetype='image/jpeg')

# Function to overlay an image on top of another at a specific location
def overlay_image(background, overlay_type, x, y, width, height):

    if overlay_type == 0 or overlay_type == 1:
        overlay_path = "assets/eye2.png"
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        resized_overlay = cv2.resize(overlay, (width, width))
        
    elif overlay_type == 2:
        overlay_path = "assets/nose.png"
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        resized_overlay = cv2.resize(overlay, (width, width))
    else:
        overlay_path = "assets/mouth.png"
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        resized_overlay = cv2.resize(overlay, (width, height))

    edited_image = cvzone.overlayPNG(background, resized_overlay, [x, y])

    return edited_image
    
    #alpha_channel = resized_overlay[:, :, 3] 

if __name__ == "__main__":
    predict("image2.jpg")