from ultralytics import YOLO
from itertools import product

def main():
    model = YOLO("yolov8s.pt")
    train_results = model.train(data='data.yaml',
        epochs=100,
        imgsz=640,
        #weight_decay=0.0005,
        device=0)

    # evaluate performace on the validation set
    metrics = model.val()
    # perform obj detection on image
    results = model("datasets/test/img1000.png")
    results[0].show()
    # export format
    export_model = model.export(format="onnx")

if __name__ == "__main__":
    main()