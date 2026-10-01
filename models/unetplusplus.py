"""Standalone UNet++ (nested UNet) model for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_components import DoubleConv, Down, initialize_weights


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, bn=False,
                 base_width=14, depth=5):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth

        filters = [base_width * (2 ** i) for i in range(depth)]

        # Encoder (column 0): inc + Down modules (same as UNet)
        self.inc = DoubleConv(n_channels, filters[0], bn=bn)
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(filters[i], filters[i + 1], bn=bn))

        # Upsampling: one bilinear + Conv2d 3×3 per decoder node (i, j).
        # Each node gets its own learnable projection (matching classical UNet++).
        self.upsample = nn.ModuleList()
        for i in range(depth - 1):
            row = nn.ModuleList()
            for j in range(1, depth - i):
                row.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(filters[i + 1], filters[i], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ))
            self.upsample.append(row)

        # Nested decoder nodes conv_{i}_{j} for j >= 1
        # After upsample: up_ch = filters[i] (channel-reduced by the Conv2d above)
        self.nested_convs = nn.ModuleList()
        for i in range(depth - 1):
            row = nn.ModuleList()
            for j in range(1, depth - i):
                in_ch = filters[i] * j + filters[i]  # j same-level outputs + upsampled
                row.append(DoubleConv(in_ch, filters[i], bn=bn))
            self.nested_convs.append(row)

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        initialize_weights(self)

    def _init(self):
        initialize_weights(self)

    def forward(self, x, return_all=False):
        # x_nodes[i][j] stores the output of node (i, j)
        x_nodes: list[list[torch.Tensor | None]] = [
            [None] * (self.depth - i) for i in range(self.depth)
        ]

        # Encoder (column 0) — uses shared Down modules
        x_nodes[0][0] = self.inc(x)
        for i in range(self.depth - 1):
            x_nodes[i + 1][0] = self.downs[i](x_nodes[i][0])

        # Nested columns j = 1 .. depth-i-1
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                dense: list[torch.Tensor] = [x_nodes[i][k] for k in range(j)]  # type: ignore[misc]
                # Upsample from level i+1 to level i
                up_row = self.upsample[i]
                assert isinstance(up_row, nn.ModuleList)
                up_feat = up_row[j - 1](x_nodes[i + 1][j - 1])
                # Pad if sizes don't match exactly
                target = x_nodes[i][0]
                assert target is not None and up_feat is not None
                diffY = target.size(2) - up_feat.size(2)
                diffX = target.size(3) - up_feat.size(3)
                if diffY != 0 or diffX != 0:
                    up_feat = F.pad(up_feat, [diffX // 2, diffX - diffX // 2,
                                              diffY // 2, diffY - diffY // 2])
                dense.append(up_feat)
                conv_row = self.nested_convs[i]
                assert isinstance(conv_row, nn.ModuleList)
                x_nodes[i][j] = conv_row[j - 1](torch.cat(dense, 1))

        final_output = self.final(x_nodes[0][self.depth - 1])
        if return_all:
            return [x_nodes[0][self.depth - 1]], final_output
        return final_output