"""
ResUNet 3D High-Quality para simulaciones TUS.
Entrada: 2 Canales (Fuente, Cráneo).
Salida:  1 Canal   (p_max_norm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Normalization helper
# -------------------------
def norm3d(ch: int, kind: str = "group", groups: int = 8):
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)

# -------------------------
# Squeeze-Excitation
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
    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n1 = norm3d(out_ch, kind=norm_kind)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
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
    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.n = norm3d(out_ch, kind=norm_kind)
        self.act = nn.SiLU(inplace=True)
        self.block = ResBlock3D(out_ch, out_ch, norm_kind=norm_kind, use_se=use_se)

    def forward(self, x):
        x = self.act(self.n(self.down(x)))
        x = self.block(x)
        return x

class UpConcat(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv_up = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block = ResBlock3D(out_ch + skip_ch, out_ch, norm_kind=norm_kind, use_se=use_se)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x

# -------------------------
# ResUNet 3D High-Quality
# -------------------------
class ResUNet3D_HQ(nn.Module):
    def __init__(
        self,
        in_ch=2,          # ¡ACTUALIZADO A 2 CANALES!
        out_ch=1,
        base=16,          
        norm_kind="group",
        use_se=True,
        out_positive=True,
    ):
        super().__init__()
        self.out_positive = out_positive
        self.out_act = nn.Softplus() if out_positive else nn.Identity()

        # Encoder
        self.stem = ResBlock3D(in_ch, base, norm_kind=norm_kind, use_se=use_se)  # 128^3
        self.d1 = Down(base, base * 2, norm_kind=norm_kind, use_se=use_se)       # 64^3
        self.d2 = Down(base * 2, base * 4, norm_kind=norm_kind, use_se=use_se)   # 32^3
        self.d3 = Down(base * 4, base * 8, norm_kind=norm_kind, use_se=use_se)   # 16^3
        self.d4 = Down(base * 8, base * 16, norm_kind=norm_kind, use_se=use_se)  # 8^3

        # Bottleneck
        self.mid = ResBlock3D(base * 16, base * 32, norm_kind=norm_kind, use_se=use_se) # 8^3

        # Decoder 
        self.u4 = UpConcat(base * 32, base * 8, base * 8, norm_kind=norm_kind, use_se=use_se) # ->16^3
        self.u3 = UpConcat(base * 8, base * 4, base * 4, norm_kind=norm_kind, use_se=use_se)  # ->32^3
        self.u2 = UpConcat(base * 4, base * 2, base * 2, norm_kind=norm_kind, use_se=use_se)  # ->64^3
        self.u1 = UpConcat(base * 2, base, base, norm_kind=norm_kind, use_se=use_se)          # ->128^3

        # Head (eliminamos self.final porque u1 ya deja los canales en 'base')
        self.head = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)   # base, 128^3
        s1 = self.d1(s0)    # 2b, 64^3
        s2 = self.d2(s1)    # 4b, 32^3
        s3 = self.d3(s2)    # 8b, 16^3
        s4 = self.d4(s3)    # 16b, 8^3

        m = self.mid(s4)    # 32b, 8^3

        x = self.u4(m, s3)  # 8b, 16^3
        x = self.u3(x, s2)  # 4b, 32^3
        x = self.u2(x, s1)  # 2b, 64^3
        x = self.u1(x, s0)  # b, 128^3 (Aquí ya se hizo el concat con s0)

        x = self.head(x)
        x = self.out_act(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Probando con 2 canales en la entrada
    model = ResUNet3D_HQ(in_ch=2, out_ch=1, base=16).to(device)
    
    # Simulamos un batch de tamaño 1
    x = torch.randn(1, 2, 128, 128, 128, device=device)
    
    with torch.no_grad():
        y = model(x)
        
    print(f"Test superado: Modelo creado y propagación forward exitosa en {device}.")
    print("Shape de salida:", tuple(y.shape))