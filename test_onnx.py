
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"]
)

image = cv2.imread("test.jpg")

if image is None:
    raise FileNotFoundError("test.jpg not found")

# Letterbox preprocessing
h, w = image.shape[:2]
scale = min(640 / w, 640 / h)

new_w = int(round(w * scale))
new_h = int(round(h * scale))

resized = cv2.resize(image, (new_w, new_h))

canvas = np.full((640, 640, 3), 114, dtype=np.uint8)

pad_x = (640 - new_w) // 2
pad_y = (640 - new_h) // 2

canvas[pad_y:pad_y + new_h,
       pad_x:pad_x + new_w] = resized

# BGR -> RGB, normalize, HWC -> CHW
image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
image_input = image_rgb.astype(np.float32) / 255.0
image_input = np.transpose(image_input, (2, 0, 1))
image_input = np.expand_dims(image_input, axis=0)

# Inference
output = session.run(
    ["output0"],
    {"images": image_input}
)[0]

predictions = output[0].T

print("Output shape:", output.shape)

for class_id, name in enumerate([
    "person_without_seatbelt",
    "person_with_seatbelt"
]):
    scores = predictions[:, 4 + class_id]

    print("\n", name)
    print("Maximum confidence:", float(scores.max()))
    print("Above 0.25:", int(np.sum(scores > 0.25)))
    print("Above 0.10:", int(np.sum(scores > 0.10)))