import torch
import onnx
import onnxruntime as ort
import numpy as np

# 1. Define and Export the Model
class BF16Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
    
    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

# Move model to bfloat16 and GPU
model = BF16Model().to(torch.bfloat16).cuda().eval()

onnx_path = "/home/shivani/work/llm-inference/src/test/model_bf16.onnx"
dummy_input = torch.randn(1, 3, dtype=torch.bfloat16, device='cuda')

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    opset_version=16, # Opset 16+ is required for robust BF16 support
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Model exported to {onnx_path}")

# 2. Setup ONNX Runtime (GPU Only)
providers = ['CUDAExecutionProvider']
session = ort.InferenceSession(onnx_path, providers=providers)
print("Success: Model loaded natively on GPU!")

# 3. Prepare Inputs and Outputs using PyTorch (Native BF16)
# We stay on 'cuda' the whole time
input_tensor = torch.randn(1, 3, dtype=torch.bfloat16, device='cuda')
output_tensor = torch.empty(1, 3, dtype=torch.bfloat16, device='cuda')

# 4. Use IO Binding to bridge PyTorch and ONNX
io_binding = session.io_binding()

# Bind the Input
io_binding.bind_input(
    name='input',
    device_type='cuda',
    device_id=0,
    element_type=16, # bfloat16 is internally handled as uint16 bits
    shape=input_tensor.shape,
    buffer_ptr=input_tensor.data_ptr()
)

# Bind the Output
io_binding.bind_output(
    name='output',
    device_type='cuda',
    device_id=0,
    element_type=16,
    shape=output_tensor.shape,
    buffer_ptr=output_tensor.data_ptr()
)

# 5. Run Inference
session.run_with_iobinding(io_binding)

print("\n--- Inference Results ---")
print(f"Input (BF16):\n{input_tensor}")
print(f"Output (BF16):\n{output_tensor}")