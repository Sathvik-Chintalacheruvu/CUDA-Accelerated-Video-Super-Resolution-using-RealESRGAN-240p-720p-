import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualDenseBlock5C(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32):
        super().__init__()
        gc = growth_channels
        self.conv1 = nn.Conv2d(in_channels, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * gc, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        # residual scaling
        return x + 0.2 * x5



class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(in_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock5C(in_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock5C(in_channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # residual scaling
        return x + 0.2 * out



class RRDBNet(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 scale=4):
        super().__init__()

        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)


        rrdb_blocks = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.body = nn.Sequential(*rrdb_blocks)

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


        assert scale in [2, 4], "This RRDBNet is configured for x2 or x4"
        upsample_layers = []
        for _ in range(int(scale / 2)):  # x2 twice -> x4
            upsample_layers += [
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # we keep them separate so their names match real weights (conv_up1, conv_up2)
        self.conv_up1 = upsample_layers[0]
        self.lrelu_up1 = upsample_layers[1]
        self.conv_up2 = upsample_layers[2]
        self.lrelu_up2 = upsample_layers[3]

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        # x2
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu_up1(self.conv_up1(fea))
        # x4
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu_up2(self.conv_up2(fea))

        fea = self.lrelu(self.conv_hr(fea))
        out = self.conv_last(fea)
        return out


# -----------------------------
#   Helper: load pretrained RealESRGAN_x4plus weights
# -----------------------------
def load_realesrgan_x4(model_path: str,
                       device: torch.device = None,
                       use_half: bool = True) -> RRDBNet:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    print(f" Loading RealESRGAN_x4plus weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # official checkpoints store weights under "params_ema" or "params"
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    if use_half and device.type == "cuda":
        model.half()

    return model
