import torch
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

try:
    import xformers
    print(f"Xformers version: {xformers.__version__}")
except ImportError:
    print("Xformers not installed")

try:
    import unsloth
    print(f"Unsloth version: {unsloth.__version__}") # Unsloth might not have __version__ exposed easily in older versions
except ImportError:
    print("Unsloth not installed")
except AttributeError:
    print("Unsloth installed but __version__ not found")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("No GPU detected")
