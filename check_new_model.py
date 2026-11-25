import onnxruntime as ort

model_path = "seatbelt_yolo11.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("MODEL LOADED SUCCESSFULLY")

print("\nINPUTS:")
for item in session.get_inputs():
    print("Name:", item.name)
    print("Shape:", item.shape)
    print("Type:", item.type)

print("\nOUTPUTS:")
for item in session.get_outputs():
    print("Name:", item.name)
    print("Shape:", item.shape)
    print("Type:", item.type)

print("\nMETADATA:")
metadata = session.get_modelmeta()
print(metadata.custom_metadata_map)