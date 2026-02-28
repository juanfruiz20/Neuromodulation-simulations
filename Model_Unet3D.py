# model_unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Normalización estable con batch pequeño
# -------------------------
def norm3d(ch: int, kind: str = "group", groups: int = 8):
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)


# -------------------------
# Squeeze-Excitation (opcional pero suele mejorar detalle)
# -------------------------
class SEBlock3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        r = max(1, ch // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(ch, r, kernel_size=1)
        self.fc2 = nn.Conv3d(r, ch, kernel_size=1)

    def forward(self, x):
        s = self.pool(x)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


# -------------------------
# Residual Block 3D
# -------------------------
class ResBlock3D(nn.Module):
    """
    Conv-Norm-Act-Conv-Norm + skip
    Opcional: SE para recalibración de canales.
    """

    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n1 = norm3d(out_ch, kind=norm_kind)
        self.conv2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n2 = norm3d(out_ch, kind=norm_kind)
        self.act = nn.SiLU(inplace=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.use_se = use_se
        self.se = SEBlock3D(out_ch, reduction=8) if use_se else nn.Identity()

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        x = self.act(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        x = self.se(x)
        x = self.act(x + identity)
        return x


# -------------------------
# Down / Up
# -------------------------
class Down(nn.Module):
    """Downsample + ResBlock."""

    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()
        # Strided conv: learnable downsampling (suele ir mejor que MaxPool)
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.n = norm3d(out_ch, kind=norm_kind)
        self.act = nn.SiLU(inplace=True)
        self.block = ResBlock3D(
            out_ch, out_ch, norm_kind=norm_kind, use_se=use_se)

    def forward(self, x):
        x = self.act(self.n(self.down(x)))
        x = self.block(x)
        return x


class UpConcat(nn.Module):
    """
    Upsample trilineal + 1x1 conv + CONCAT skip + ResBlock
    (Más calidad que suma, pero consume más VRAM)
    """

    def __init__(self, in_ch, skip_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False)
        self.conv_up = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block = ResBlock3D(out_ch + skip_ch, out_ch,
                                norm_kind=norm_kind, use_se=use_se)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_up(x)

        # seguridad por si hubiera mismatch
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(
                x, size=skip.shape[-3:], mode="trilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


# -------------------------
# ResUNet 3D High-Quality
# -------------------------
class ResUNet3D_HQ(nn.Module):
    """
    Input : [B, 4, 128,128,128]
    Output: [B, 1, 128,128,128] (p_max_norm >= 0)

    Ajustes clave para "calidad del haz":
    - concat skips (más info en decoder)
    - residual + SE (mejor detalle)
    - upsample trilineal (evita checkerboard)
    - salida positiva con softplus
    """

    def __init__(
        self,
        in_ch=4,
        out_ch=1,
        base=16,                 # prueba 16; si cabe, sube a 24
        norm_kind="group",
        use_se=True,
        out_positive=True,
    ):
        super().__init__()
        self.out_positive = out_positive
        self.out_act = nn.Softplus()

        # Encoder
        self.stem = ResBlock3D(
            in_ch, base, norm_kind=norm_kind, use_se=use_se)          # 128^3
        self.d1 = Down(base, base * 2, norm_kind=norm_kind,
                       use_se=use_se)              # 64^3
        self.d2 = Down(base * 2, base * 4, norm_kind=norm_kind,
                       use_se=use_se)          # 32^3
        self.d3 = Down(base * 4, base * 8, norm_kind=norm_kind,
                       use_se=use_se)          # 16^3
        self.d4 = Down(base * 8, base * 16, norm_kind=norm_kind,
                       use_se=use_se)         # 8^3

        # Bottleneck (más capacidad donde el coste espacial es pequeño)
        self.mid = ResBlock3D(base * 16, base * 32,
                              norm_kind=norm_kind, use_se=use_se)  # 8^3

        # Decoder (concat skips)
        self.u4 = UpConcat(base * 32, skip_ch=base * 16, out_ch=base *
                           16, norm_kind=norm_kind, use_se=use_se)  # ->16^3
        self.u3 = UpConcat(base * 16, skip_ch=base * 8,  out_ch=base * 8,
                           norm_kind=norm_kind, use_se=use_se)  # ->32^3
        self.u2 = UpConcat(base * 8,  skip_ch=base * 4,  out_ch=base * 4,
                           norm_kind=norm_kind, use_se=use_se)  # ->64^3
        self.u1 = UpConcat(base * 4,  skip_ch=base * 2,  out_ch=base * 2,
                           norm_kind=norm_kind, use_se=use_se)  # ->128^3

        # Fusión final con stem (concat una última vez para máximo detalle)
        self.final = ResBlock3D(base * 2 + base, base,
                                norm_kind=norm_kind, use_se=use_se)

        # Head
        self.head = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)   # base, 128^3
        s1 = self.d1(s0)    # 2b, 64^3
        s2 = self.d2(s1)    # 4b, 32^3
        s3 = self.d3(s2)    # 8b, 16^3
        s4 = self.d4(s3)    # 16b, 8^3

        m = self.mid(s4)    # 32b, 8^3

        x = self.u4(m, s4)  # 16b, 16^3
        x = self.u3(x, s3)  # 8b, 32^3
        x = self.u2(x, s2)  # 4b, 64^3
        x = self.u1(x, s1)  # 2b, 128^3

        x = torch.cat([x, s0], dim=1)
        x = self.final(x)

        x = self.head(x)
        if self.out_positive:
            x = self.out_act(x)
        return x


# Mini test opcional
"""if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResUNet3D_HQ(in_ch=4, out_ch=1, base=16, norm_kind="group", use_se=True).to(device)
    x = torch.randn(1, 4, 128, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print("out:", y.shape, "min/max:", float(y.min()), float(y.max()))"""
