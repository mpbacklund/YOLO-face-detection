import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

def predict(image):
    # Load the test image
    image = cv2.imread(image)

    # Perform inference
    results = model(image)

    # Process each detection
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
        classes = result.boxes.cls.cpu().numpy()  # Class indices
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)  # Get box coordinates
            width, height = x2 - x1, y2 - y1  # Calculate box dimensions

            overlay_image(image, cls, x1, y1, width, height)

    output_scale = 0.1  # Scale factor (e.g., 0.5 = 50% smaller)
    new_width = int(image.shape[1] * output_scale)
    new_height = int(image.shape[0] * output_scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save and/or display the resized image
    output_path = "output_with_overlays_resized.jpg"
    cv2.imwrite(output_path, resized_image)  # Save the resized image
    cv2.imshow("Resized Result", resized_image)  # Show the resized image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_and_resize_overlay():
    

# Function to overlay an image on top of another at a specific location
def overlay_image(background, overlay_type, x, y, width, height):
    # decide what image to overlay and load it
    if overlay_type == 0 or overlay_type == 1:
        overlay_path = "assets/eye.png"
    elif overlay_type == 2:
        overlay_path = "assets/nose.png"
    else:
        overlay_path = "assets/mouth.png"
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if exists

    # Resize overlay to the bounding box size
    resized_overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)

    # If overlay has an alpha channel, handle transparency
    if resized_overlay.shape[2] == 4:  # Check if overlay has alpha channel
        alpha = resized_overlay[:, :, 3] / 255.0
        for c in range(0, 3):  # Loop through color channels
            background[y:y+height, x:x+width, c] = (
                alpha * resized_overlay[:, :, c] +
                (1 - alpha) * background[y:y+height, x:x+width, c]
            )
    else:  # If no alpha channel, just replace
        background[y:y+height, x:x+width] = resized_overlay

if __name__ == "__main__":
    predict("image.jpg")