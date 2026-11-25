from ultralytics import YOLO
from pathlib import Path

model = YOLO("seatbeltWbest (1).pt")

folders = {
    "with_belt": "seatbelt_dataset/with_belt",
    "without_belt": "seatbelt_dataset/without_belt"
}

for expected, folder in folders.items():
    print(f"\n===== {expected} =====")

    for image in sorted(Path(folder).glob("*")):
        if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        results = model.predict(
            source=str(image),
            imgsz=640,
            conf=0.10,
            verbose=False
        )

        detections = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append(
                f"{model.names[cls]} ({conf:.2f})"
            )

        print(image.name, "=>", detections)