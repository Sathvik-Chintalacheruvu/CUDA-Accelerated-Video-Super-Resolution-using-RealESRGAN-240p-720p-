import torch

pth = "models/RealESRGAN_x4plus.pth"
ckpt = torch.load(pth, map_location="cpu")
state = ckpt["params_ema"]

print("conv_first:", state["conv_first.weight"].shape)
print("has NaN:", torch.isnan(state["conv_first.weight"]).any())
