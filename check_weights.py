import torch

ckpt = torch.load("models/RealESRGAN_x2plus.pth", map_location="cpu")
state = ckpt["params_ema"]

print("conv_first.weight shape:", state["conv_first.weight"].shape)
