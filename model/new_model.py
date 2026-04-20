"""
UNet3D and PatchDiscriminator3D with configurable input channels.

Extends model.py so that the generator accepts multi-channel (e.g. 3-channel
vector displacement) inputs and produces multi-channel outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Spectral norm helper (copied from model.py) ──────────────────────────────

class _CPUSpectralNormConv3d(nn.Module):
    """Conv3d with spectral norm estimated by CPU power iteration."""

    def __init__(self, conv: nn.Conv3d, n_power_iterations: int = 1, eps: float = 1e-12):
        super().__init__()
        self.conv = conv
        self.n_power_iterations = max(1, int(n_power_iterations))
        self.eps = float(eps)
        self.register_buffer('_sn_u', torch.empty(0, dtype=torch.float32), persistent=False)

    def _reshape_weight_to_matrix(self, w: torch.Tensor) -> torch.Tensor:
        return w.reshape(w.shape[0], -1)

    def _compute_sigma(self, w: torch.Tensor) -> torch.Tensor:
        w_cpu = self._reshape_weight_to_matrix(w).detach().float().cpu()
        out_features = w_cpu.shape[0]
        if self._sn_u.numel() != out_features:
            u0 = torch.randn(out_features, dtype=torch.float32)
            self._sn_u = F.normalize(u0, dim=0, eps=self.eps)
        u = self._sn_u
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(w_cpu.t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.mv(w_cpu, v), dim=0, eps=self.eps)
        self._sn_u = u.detach()
        sigma = torch.dot(u, torch.mv(w_cpu, v)).clamp_min(self.eps)
        return sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._compute_sigma(self.conv.weight)
        sigma = sigma.to(device=self.conv.weight.device, dtype=self.conv.weight.dtype)
        w_bar = self.conv.weight / sigma
        return F.conv3d(
            x, w_bar, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups,
        )


# ── UNet3D building blocks ───────────────────────────────────────────────────

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,
            diffH // 2, diffH - diffH // 2,
            diffD // 2, diffD - diffD // 2,
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ── UNet3D with configurable in_channels ──────────────────────────────────────

class UNet3D(nn.Module):
    """3D U-Net with configurable input channels and depth."""

    def __init__(self, n_classes, in_channels=3, trilinear=True,
                 base_channels=16, num_pools=5):
        super().__init__()
        if not (1 <= num_pools <= 5):
            raise ValueError(f"num_pools must be in [1, 5], got {num_pools}")

        self.n_classes = n_classes
        self.in_channels = in_channels
        self.trilinear = trilinear
        self.num_pools = num_pools

        factor = 2 if trilinear else 1

        self.inc = DoubleConv3D(in_channels, base_channels)

        self.feature_channels = [base_channels]
        self.downs = nn.ModuleList()
        ch_in = base_channels
        for level in range(1, num_pools + 1):
            if level < num_pools:
                ch_out = base_channels * (2 ** level)
            else:
                ch_out = (base_channels * (2 ** level)) // factor
            self.downs.append(Down3D(ch_in, ch_out))
            self.feature_channels.append(ch_out)
            ch_in = ch_out

        self.ups = nn.ModuleList()
        decoder_ch = self.feature_channels[-1]
        for level in range(num_pools - 1, -1, -1):
            skip_ch = self.feature_channels[level]
            up_in = decoder_ch + skip_ch
            up_out = skip_ch if level == 0 else (skip_ch // factor)
            self.ups.append(Up3D(up_in, up_out, trilinear))
            decoder_ch = up_out

        self.outc = OutConv3D(base_channels, n_classes)

    def forward(self, x):
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        x = features[-1]
        for idx, up in enumerate(self.ups):
            skip = features[-(idx + 2)]
            x = up(x, skip)
        return self.outc(x)


# ── PatchDiscriminator3D with configurable in_channels ────────────────────────

class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator with configurable input channels.

    For vector fields: in_channels = 2 * n_components (condition + prediction
    concatenated along channel dim), e.g. 6 for 3-channel psi.
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 32,
                 n_layers: int = 3, use_spectral_norm: bool = True,
                 spectral_norm_impl: str = 'native'):
        super().__init__()

        if spectral_norm_impl not in ('native', 'cpu_power_iter'):
            raise ValueError(
                f"spectral_norm_impl must be 'native' or 'cpu_power_iter', "
                f"got {spectral_norm_impl}"
            )

        def sn(module: nn.Module) -> nn.Module:
            if not use_spectral_norm:
                return module
            if spectral_norm_impl == 'cpu_power_iter':
                if not isinstance(module, nn.Conv3d):
                    raise TypeError('cpu_power_iter spectral norm only supports Conv3d')
                return _CPUSpectralNormConv3d(module)
            return nn.utils.parametrizations.spectral_norm(module)

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            sn(nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)),
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

        self.final = nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, ic: torch.Tensor, residual: torch.Tensor,
                return_features: bool = False):
        x = torch.cat([ic, residual], dim=1)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        out = self.final(x)
        if return_features:
            return out, features
        return out


# ── MultiScaleDiscriminator3D ─────────────────────────────────────────────────

class MultiScaleDiscriminator3D(nn.Module):
    """Multi-scale 3D PatchGAN discriminator (pix2pixHD style).

    Sub-discriminators (all at the same spatial resolution):
    - D_fine   (n_layers=1, stride=2) → large output map (~10³ for 20³ input),
                                        sensitive to local high-frequency structure.
    - D_coarse (n_layers=n_layers_coarse, stride=2) → small output map (~3³),
                                        captures global patch-level structure.

    forward() returns:
      return_features=False : list of logit tensors
      return_features=True  : list of (logits, features_list) tuples
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 64,
                 n_layers_coarse: int = 3, use_spectral_norm: bool = False,
                 spectral_norm_impl: str = 'native'):
        super().__init__()
        discs = [
            PatchDiscriminator3D(
                in_channels=in_channels, base_channels=base_channels,
                n_layers=1, use_spectral_norm=use_spectral_norm,
                spectral_norm_impl=spectral_norm_impl,
            ),  # D_fine
            PatchDiscriminator3D(
                in_channels=in_channels, base_channels=base_channels,
                n_layers=n_layers_coarse, use_spectral_norm=use_spectral_norm,
                spectral_norm_impl=spectral_norm_impl,
            ),  # D_coarse
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, ic: torch.Tensor, residual: torch.Tensor,
                return_features: bool = False):
        return [d(ic, residual, return_features=return_features)
                for d in self.discriminators]
