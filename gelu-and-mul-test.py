import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import use_kernel_forward_from_hub
from kernels import use_kernel_mapping, LayerRepository
from kernels import Mode, kernelize

# Define the hub kernel to use for this test on cuda devices
kernel_layer_mapping = {
    "GeluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="GeluAndMul",
            version=1,
        )
    }
}

# Implement the torch fallback method and request to use a hub kernel if available
@use_kernel_forward_from_hub("GeluAndMul")
class GeluAndMul(nn.Module):
    """Implementation from https://github.com/vllm-project/vllm
       vllm/model_executor/layers/activation.py
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate="none") * x[..., d:]

# Run the pure torch method first so we can compare the hub kernel
x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
model = GeluAndMul()
torch_out = model(x)
hub_out = None

# Run the hub optimized kernel now
with use_kernel_mapping(kernel_layer_mapping):
    # Tell kernels that we want to do inference and enable torch.compile
    model = kernelize(model, device="cuda", mode=Mode.INFERENCE | Mode.TORCH_COMPILE)

    hub_out = model(x)

# Make sure the hub optimized kernel gives the same output
if torch.allclose(hub_out, torch_out, atol=1e-3, rtol=1e-3):
    print("Success!")
else:
    print("Failed...")
