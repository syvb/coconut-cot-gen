import os, sys
os.environ.setdefault("PJRT_DEVICE", "TPU")
import torch, torch_xla
import torch_xla.core.xla_model as xm
print("torch:", torch.__version__, flush=True)
print("xla:", torch_xla.__version__, flush=True)
print("supported devices:", xm.get_xla_supported_devices(), flush=True)
dev = xm.xla_device()
print("device:", dev, flush=True)
x = torch.randn(3, 3, device=dev)
print("matmul sum:", (x @ x).sum().item(), flush=True)
print("OK", flush=True)
