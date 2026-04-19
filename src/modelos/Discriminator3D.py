import torch
import torch.nn as nn


class PatchDiscriminator3D(nn.Module):
    """
    3D Conditional PatchGAN Discriminator.

    This discriminator classifies overlapping patches of generated vs. real acoustic fields.
    It uses a conditional architecture that takes both the input conditions and the acoustic
    field to make predictions.

    Args:
        in_ch (int): Number of input channels (default: 3). Typically concatenates:
                     - source mask/conditions (2 channels)
                     - acoustic field (1 channel)
        base (int): Number of base filters (default: 16). Used to scale channel dimensions
                    throughout the network.

    Inputs during forward pass:
        x (torch.Tensor): Conditional input [B, Cx, D, H, W]
                         Example: source_mask or skull geometry + other conditions
        field (torch.Tensor): Acoustic field [B, 1, D, H, W]
                             Real or fake field to discriminate

    Output:
        torch.Tensor: Patch-wise classification scores [B, 1, d', h', w']
                     Each spatial location represents the discriminator's confidence
                     for the patch centered at that location.
    """

    def __init__(self, in_ch: int = 3, base: int = 16):
        super().__init__()

        # Build convolutional network with progressive downsampling
        # Using InstanceNorm instead of BatchNorm for better performance with small batch sizes
        self.net = nn.Sequential(
            # Layer 1: Initial convolution with stride=2 (downsample by 2x)
            # Input: [B, in_ch, D, H, W] -> Output: [B, base, D/2, H/2, W/2]
            nn.Conv3d(in_ch, base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: Downsample by 2x again, double feature channels
            # Input: [B, base, D/2, H/2, W/2] -> Output: [B, base*2, D/4, H/4, W/4]
            nn.Conv3d(base, base * 2, kernel_size=4, stride=2, padding=1),
            # Normalize per sample (better for GANs)
            nn.InstanceNorm3d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: Downsample by 2x once more, double feature channels again
            # Input: [B, base*2, D/4, H/4, W/4] -> Output: [B, base*4, D/8, H/8, W/8]
            nn.Conv3d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: Keep spatial dimensions fixed (stride=1), expand feature channels
            # Input: [B, base*4, D/8, H/8, W/8] -> Output: [B, base*8, D/8, H/8, W/8]
            nn.Conv3d(base * 4, base * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm3d(base * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: Final convolution to single output channel (discriminator scores)
            # Input: [B, base*8, D/8, H/8, W/8] -> Output: [B, 1, D/8, H/8, W/8]
            # This produces patch-wise predictions (PatchGAN approach)
            nn.Conv3d(base * 8, 1, kernel_size=4, stride=1, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize network weights using the technique from pix2pix paper.
        This improves training stability and convergence of GANs.

        - Convolutional weights: Normal distribution (mean=0, std=0.02)
        - Biases: Zeros
        - Normalization layers: weights ~ N(1, 0.02), biases = 0
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # Initialize convolution kernels with small random values
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                # Initialize normalization layer weights around 1 and biases to 0
                if m.weight is not None:
                    nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Conditional input [B, Cx, D, H, W]
                             (e.g., source mask, skull geometry)
            field (torch.Tensor): Acoustic field [B, 1, D, H, W]
                                 (real or generated)

        Returns:
            torch.Tensor: Patch-wise discriminator predictions [B, 1, D/8, H/8, W/8]
                         High values indicate "real", low values indicate "fake"
        """
        # Concatenate condition and field along channel dimension
        z = torch.cat([x, field], dim=1)
        # Pass through discriminator network
        return self.net(z)
