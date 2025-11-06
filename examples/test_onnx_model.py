import onnxruntime as ort
import numpy as np

# Load the model
session = ort.InferenceSession("export3-8B/qwen.onnx")

# Print input info
print("Model inputs:")
for inp in session.get_inputs():
    print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("\nModel outputs:")
for out in session.get_outputs():
    print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
