"""Standalone UNet 3+ model with full-scale skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_components import DoubleConv, Down, initialize_weights

class UNet3Plus(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, bn=False,
                 base_width=17, depth=5):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._bn = bn
        self.depth = depth

        filters = [base_width * (2 ** i) for i in range(depth)]

        # Encoder: inc + Down modules (same as UNet)
        self.inc = DoubleConv(n_channels, filters[0], bn=bn)
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(filters[i], filters[i + 1], bn=bn))

        # Decoder: full-scale skip connections
        self.CatChannels = filters[0]
        self.CatBlocks = depth
        self.UpChannels = self.CatChannels * self.CatBlocks

        # For each decoder stage d (from depth-2 down to 0),
        # every encoder level and every deeper decoder level contributes.
        self.skip_connections = nn.ModuleList()  # [depth-1] stages
        self.decoder_convs = nn.ModuleList()
        for d in range(depth - 2, -1, -1):
            stage_skips = nn.ModuleList()
            for src in range(depth):
                if src < d:
                    # Encoder level src -> pool down to level d
                    scale = 2 ** (d - src)
                    stage_skips.append(self._make_skip_connection(filters[src], scale))
                elif src == d:
                    # Same-level encoder -> identity
                    stage_skips.append(self._make_skip_connection(filters[src], 1))
                elif src == depth - 1:
                    # Bottleneck encoder -> upsample
                    scale = -(2 ** (src - d))
                    stage_skips.append(self._make_skip_connection(filters[src], scale))
                else:
                    # Deeper decoder stage -> upsample
                    scale = -(2 ** (src - d))
                    stage_skips.append(self._make_skip_connection(self.UpChannels, scale))
            self.skip_connections.append(stage_skips)
            self.decoder_convs.append(self._make_decoder_conv(self.UpChannels, self.UpChannels))

        # Output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, kernel_size=1)

        initialize_weights(self)

    def _init(self):
        initialize_weights(self)

    def _make_decoder_conv(self, in_channels, out_channels):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if self._bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_skip_connection(self, in_channels, scale_factor):
        layers: list[nn.Module] = []
        if scale_factor > 1:  # Downsample via MaxPool
            layers.append(nn.MaxPool2d(scale_factor, scale_factor, ceil_mode=True))
        elif scale_factor < 0:  # Upsample (parameter-free resize; the Conv2d below learns the projection)
            factor = abs(scale_factor)
            layers.append(nn.Upsample(scale_factor=factor, mode='bilinear'))
        # Identity case (scale_factor == 1): no resize layer
        layers.append(nn.Conv2d(in_channels, self.CatChannels, 3, padding=1))
        if self._bn:
            layers.append(nn.BatchNorm2d(self.CatChannels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, return_all=False):
        # Encoder — uses shared Down modules
        enc = [self.inc(x)]
        for down in self.downs:
            enc.append(down(enc[-1]))

        # Decoder (from deepest decoder stage to shallowest)
        dec: dict[int, torch.Tensor] = {}
        for stage_idx, d in enumerate(range(self.depth - 2, -1, -1)):
            cat_parts: list[torch.Tensor] = []
            stage = self.skip_connections[stage_idx]
            assert isinstance(stage, nn.ModuleList)
            for src in range(self.depth):
                skip_mod = stage[src]
                if src <= d or src == self.depth - 1:
                    cat_parts.append(skip_mod(enc[src]))
                else:
                    cat_parts.append(skip_mod(dec[src]))
            dec[d] = self.decoder_convs[stage_idx](torch.cat(cat_parts, 1))

        logits = self.outconv1(dec[0])
        if return_all:
            return [dec[0]], logits
        return logits