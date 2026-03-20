import torch
import torch.nn as nn
import torch.nn.functional as F
# ──────────────────────────────────────────────────────────────────────────────
# UNet3D (generator / standalone regressor)
# ──────────────────────────────────────────────────────────────────────────────


class DoubleConv3D(nn.Module):
    """3D double-convolution block: (Conv3D => BN => ReLU) * 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """3D downsampling: MaxPool3D + double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """3D upsampling: transposed conv/interp + skip connection + double conv."""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # Upsample with trilinear interpolation or transposed convolution.
        if trilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle spatial-size mismatch before concatenation.
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,
            diffH // 2, diffH - diffH // 2,
            diffD // 2, diffD - diffD // 2
        ])

        # Skip connection via channel-wise concatenation.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """3D output convolution: map channels to classes using 1x1x1 conv."""

    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net with configurable depth via number of pooling layers."""

    def __init__(self, n_classes, trilinear=True, base_channels=16, num_pools=5):
        """
        Args:
            n_classes: Number of output channels.
            trilinear: Whether to use trilinear interpolation for upsampling.
            base_channels: Base channel width controlling model capacity.
            num_pools: Number of downsampling pooling levels (1-5).
        """
        super(UNet3D, self).__init__()
        if not (1 <= num_pools <= 5):
            raise ValueError(f"num_pools must be in [1, 5], got {num_pools}")

        self.n_classes = n_classes
        self.trilinear = trilinear
        self.num_pools = num_pools

        factor = 2 if trilinear else 1

        # Encoder (input channels = 1).
        self.inc = DoubleConv3D(1, base_channels)

        # Feature channels by level: c0 (inc), c1..c_num_pools (down outputs).
        self.feature_channels = [base_channels]
        self.downs = nn.ModuleList()

        in_channels = base_channels
        for level in range(1, num_pools + 1):
            if level < num_pools:
                out_channels = base_channels * (2 ** level)
            else:
                out_channels = (base_channels * (2 ** level)) // factor
            self.downs.append(Down3D(in_channels, out_channels))
            self.feature_channels.append(out_channels)
            in_channels = out_channels

        # Decoder (mirrored structure).
        self.ups = nn.ModuleList()
        decoder_channels = self.feature_channels[-1]
        for level in range(num_pools - 1, -1, -1):
            skip_channels = self.feature_channels[level]
            up_in_channels = decoder_channels + skip_channels
            up_out_channels = skip_channels if level == 0 else (skip_channels // factor)
            self.ups.append(Up3D(up_in_channels, up_out_channels, trilinear))
            decoder_channels = up_out_channels

        # Output layer.
        self.outc = OutConv3D(base_channels, n_classes)

    def forward(self, x):
        # Input x shape: [batch, 1, D, H, W]
        features = [self.inc(x)]

        for down in self.downs:
            features.append(down(features[-1]))

        x = features[-1]
        for idx, up in enumerate(self.ups):
            skip = features[-(idx + 2)]
            x = up(x, skip)

        logits = self.outc(x)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# PatchDiscriminator3D (conditional GAN discriminator)
# ──────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator conditioned on IC patch.

    Input : (IC_patch, residual_patch) concatenated → [B, 2, D, H, W]
    Output: spatial field of real/fake logits      → [B, 1, d, h, w]

    For patch_size=20 and n_layers=3 the output is ≈ 3×3×3.
    Uses InstanceNorm + LeakyReLU, no sigmoid (compatible with LSGAN /
    BCEWithLogitsLoss). The no-sigmoid design lets the caller choose the
    loss function freely.

    Args:
        use_spectral_norm: wrap every Conv3d with spectral normalisation,
                           which bounds the Lipschitz constant of D and
                           prevents the discriminator from dominating too
                           quickly during adversarial training.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32,
                 n_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm if use_spectral_norm else (lambda m: m)

        # Store intermediate-feature blocks as a ModuleList so that
        # forward() can optionally return their activations for feature
        # matching.
        self.blocks = nn.ModuleList()

        # First block — no normalisation layer
        self.blocks.append(nn.Sequential(
            sn(nn.Conv3d(in_channels, base_channels,
                         kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        ch = base_channels
        for _ in range(1, n_layers):
            ch_out = min(ch * 2, 256)
            self.blocks.append(nn.Sequential(
                sn(nn.Conv3d(ch, ch_out, kernel_size=3, stride=2, padding=1)),
                nn.InstanceNorm3d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = ch_out

        # Final 1-channel classification layer — no spectral norm.
        self.final = nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, ic: torch.Tensor, residual: torch.Tensor,
                return_features: bool = False):
        """Forward pass.

        Args:
            ic:              IC density patch  [B, 1, D, H, W]
            residual:        Residual patch     [B, 1, D, H, W]
            return_features: If True, also return list of intermediate
                             block activations (used for feature matching).

        Returns:
            logits (always), and optionally a list of feature tensors.
        """
        x = torch.cat([ic, residual], dim=1)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        out = self.final(x)
        if return_features:
            return out, features
        return out