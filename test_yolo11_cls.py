
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "seatbelt_yolo11.onnx"

CLASS_NAMES = ["no_seatbelt", "seat_belt"]

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

def preprocess(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize while preserving aspect ratio
    h, w = image.shape[:2]
    scale = 256 / min(h, w)
    new_w = round(w * scale)
    new_h = round(h * scale)

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Center crop: 224x224
    start_x = (new_w - 224) // 2
    start_y = (new_h - 224) // 2

    image = image[
        start_y:start_y + 224,
        start_x:start_x + 224
    ]
    cv2.imwrite(
    "debug_" + image_path,
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
)

    # HWC -> CHW, normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image


def predict(image_path):
    input_tensor = preprocess(image_path)

    output = session.run(
        ["output0"],
        {input_name: input_tensor}
    )[0][0]

    print(f"\nImage: {image_path}")
    for name, probability in zip(CLASS_NAMES, output):
        print(f"{name}: {probability * 100:.2f}%")

    predicted_class = int(np.argmax(output))
    print("Prediction:", CLASS_NAMES[predicted_class])


for path in ["test.jpg", "test1.jpg"]:
    predict(path)