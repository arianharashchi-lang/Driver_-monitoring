
import onnx
import onnxruntime as ort

model_path = "best.onnx"

# Check model structure
model = onnx.load(model_path)
onnx.checker.check_model(model)

print("Model structure: OK")

# Create inference session
session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

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

print("\nMetadata:")
print(model.metadata_props)
