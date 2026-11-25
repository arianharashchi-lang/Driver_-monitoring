
from ultralytics import YOLO

model = YOLO("best.pt")

images = ["test.jpg", "test1.jpg"]

for image_path in images:
    print("\n========================")
    print("Testing:", image_path)
    print("========================")

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.001,
        save=True
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                "Class:", class_id,
                "| Name:", model.names[class_id],
                "| Confidence:", round(confidence, 4)
            )