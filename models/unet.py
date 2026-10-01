"""Standalone UNet model for semantic segmentation."""

import torch.nn as nn

from models.model_components import DoubleConv, Down, Up, initialize_weights


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bn=False,
                 base_width=15, depth=5):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth

        filters = [base_width * (2 ** i) for i in range(depth)]

        self.inc = DoubleConv(n_channels, filters[0], bn=bn)
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(filters[i], filters[i + 1], bn=bn))
        self.ups = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = filters[depth - 1 - i]
            out_ch = filters[depth - 2 - i] if i < depth - 2 else filters[0]
            self.ups.append(Up(in_ch, out_ch, bn=bn))
        self.outc = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        initialize_weights(self)

    def _init(self):
        initialize_weights(self)
    
    def forward(self, x, return_all=False):
        enc = [self.inc(x)]
        for down in self.downs:
            enc.append(down(enc[-1]))
        x = enc[-1]
        for i, up in enumerate(self.ups):
            x = up(x, enc[self.depth - 2 - i])
        pre_logits = x
        logits = self.outc(pre_logits)
        if return_all:
            return [pre_logits], logits
        return logits