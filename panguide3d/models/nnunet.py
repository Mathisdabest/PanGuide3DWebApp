import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)

class NNUNet3D(nn.Module):
    """
    nnU-Net-ish 3D U-Net (down+up with skips).
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, levels=5, max_ch=320):
        super().__init__()
        chs = [min(max_ch, base * (2 ** i)) for i in range(levels)]
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.up   = nn.ModuleList()
        self.upc  = nn.ModuleList()

        prev = in_ch
        for c in chs:
            self.down.append(ConvBlock(prev, c))
            self.pool.append(nn.Conv3d(c, c, kernel_size=2, stride=2))
            prev = c

        bottleneck_ch = min(max_ch, chs[-1] * 2)
        self.bottleneck = ConvBlock(chs[-1], bottleneck_ch)

        for i in reversed(range(levels)):
            in_c = (bottleneck_ch if i == levels - 1 else chs[i + 1])
            self.up.append(nn.ConvTranspose3d(in_c, chs[i], kernel_size=2, stride=2))
            self.upc.append(ConvBlock(chs[i]*2, chs[i]))

        self.head = nn.Conv3d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for blk, pool in zip(self.down, self.pool):
            x = blk(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, blk, skip in zip(self.up, self.upc, reversed(skips)):
            x = up(x)
            # pad if needed due to odd sizes
            if x.shape[-3:] != skip.shape[-3:]:
                dz = skip.shape[-3] - x.shape[-3]
                dy = skip.shape[-2] - x.shape[-2]
                dx = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0,dx, 0,dy, 0,dz))
            x = torch.cat([x, skip], dim=1)
            x = blk(x)

        return self.head(x)


class NNUNet3DDeepSupervision(nn.Module):
    """
    nnU-Net-ish 3D U-Net with deep supervision heads.
    """
    def __init__(self, in_ch=1, out_ch=3, base=32, levels=5, max_ch=320):
        super().__init__()
        chs = [min(max_ch, base * (2 ** i)) for i in range(levels)]
        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()
        self.upc  = nn.ModuleList()
        self.heads = nn.ModuleList()

        # Encoder: first stage no downsample, next stages use stride-2 in first conv
        prev = in_ch
        for i, c in enumerate(chs):
            stride = 1 if i == 0 else 2
            self.down.append(ConvBlock(prev, c, stride=stride))
            prev = c

        bottleneck_ch = min(max_ch, chs[-1] * 2)
        self.bottleneck = ConvBlock(chs[-1], bottleneck_ch, stride=1)

        for i in reversed(range(levels)):
            if i == levels - 1:
                in_c = bottleneck_ch
            else:
                in_c = chs[i + 1]
            self.up.append(nn.ConvTranspose3d(in_c, chs[i], kernel_size=2, stride=2))
            self.upc.append(ConvBlock(chs[i] * 2, chs[i], stride=1))
            self.heads.append(nn.Conv3d(chs[i], out_ch, 1))

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)

        x = self.bottleneck(x)

        outputs = []
        for up, blk, skip, head in zip(self.up, self.upc, reversed(skips), self.heads):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                dz = skip.shape[-3] - x.shape[-3]
                dy = skip.shape[-2] - x.shape[-2]
                dx = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, dx, 0, dy, 0, dz))
            x = torch.cat([x, skip], dim=1)
            x = blk(x)
            outputs.append(head(x))

        # outputs from coarse->fine; return fine-first for convenience
        return list(reversed(outputs))
