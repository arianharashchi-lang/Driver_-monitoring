import cv2

# اتصال به دوربین
cap = cv2.VideoCapture(0)

# ساخت تشخیص‌دهنده انسان
hog = cv2.HOGDescriptor()
hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error!")
        break

    # تشخیص انسان
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # رسم کادر دور انسان‌ها
    for (x, y, w, h) in boxes:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # نمایش STOP در صورت تشخیص انسان
    if len(boxes) > 0:
        cv2.putText(
            frame,
            "STOP",
            (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4
        )

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()