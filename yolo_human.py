
import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

detection_history = deque(maxlen=5)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error!")
        break

    results = model.track(
    frame,
    classes=[0],
    conf=0.45,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)

    frame_height, frame_width = frame.shape[:2]
    people_count = 0
    near_detected = False
    medium_detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            people_count += 1

            person_height = y2 - y1
            height_ratio = person_height / frame_height

            if height_ratio > 0.65:
                distance_status = "VERY CLOSE"
            elif height_ratio > 0.25:
                distance_status = "MEDIUM"
            else:
                distance_status = "FAR"
            if distance_status == "VERY CLOSE":
                near_detected = True
            elif distance_status == "MEDIUM":
                medium_detected = True
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"ID: {track_id} | {confidence * 100:.1f}%"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                distance_status,
                (x1, min(y2 + 25, frame_height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    detection_history.append(near_detected)
    stop_alert = sum(detection_history) >= 3
    caution_alert = medium_detected and not stop_alert
    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    if stop_alert:
        cv2.putText(
            frame,
            "STOP",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4
        )
    elif caution_alert:
        cv2.putText(
        frame,
        "CAUTION",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        4
    )
    cv2.imshow("YOLO Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()