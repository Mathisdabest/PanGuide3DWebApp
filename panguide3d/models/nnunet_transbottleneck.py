import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nnunet import ConvBlock


def _best_head_count(embed_dim, max_heads):
    max_heads = int(max_heads)
    max_heads = min(max_heads, embed_dim)
    for h in range(max_heads, 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


def _sincos_3d_posenc(d, h, w, dim, device, dtype):
    dim_each = dim // 6
    if dim_each < 1:
        # Fallback: no positional encoding if dim too small.
        return torch.zeros(1, d * h * w, dim, device=device, dtype=dtype)

    def _axis_pos(n, dim_axis):
        pos = torch.arange(n, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(dim_axis, device=device, dtype=dtype) / dim_axis))
        sinusoid = torch.einsum("i,j->ij", pos, inv_freq)
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)  # (n, 2*dim_axis)

    emb_d = _axis_pos(d, dim_each)  # (D, 2*dim_each)
    emb_h = _axis_pos(h, dim_each)  # (H, 2*dim_each)
    emb_w = _axis_pos(w, dim_each)  # (W, 2*dim_each)

    ed = emb_d[:, None, None, :].expand(d, h, w, -1)
    eh = emb_h[None, :, None, :].expand(d, h, w, -1)
    ew = emb_w[None, None, :, :].expand(d, h, w, -1)
    pos = torch.cat([ed, eh, ew], dim=-1)  # (D, H, W, 6*dim_each)

    if pos.shape[-1] < dim:
        pad = dim - pos.shape[-1]
        pos = F.pad(pos, (0, pad))

    pos = pos.reshape(1, d * h * w, dim)
    return pos


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y)
        return x + y


class TransformerBottleneck3D(nn.Module):
    def __init__(
        self,
        channels,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        posenc_type="sincos3d",
    ):
        super().__init__()
        self.channels = int(channels)
        self.depth = int(depth)
        self.posenc_type = str(posenc_type)
        heads = _best_head_count(self.channels, num_heads)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.channels, heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(self.depth)]
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, L, C)
        if self.posenc_type == "sincos3d":
            pos = _sincos_3d_posenc(d, h, w, c, device=x.device, dtype=x.dtype)
            tokens = tokens + pos
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class NNUNetTransBottleneck3D(nn.Module):
    """
    nnU-Net-ish 3D U-Net with transformer bottleneck.
    """
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        base=32,
        levels=5,
        max_ch=320,
        trans_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        posenc_type="sincos3d",
    ):
        super().__init__()
        chs = [min(max_ch, base * (2 ** i)) for i in range(levels)]
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.up = nn.ModuleList()
        self.upc = nn.ModuleList()

        prev = in_ch
        for c in chs:
            self.down.append(ConvBlock(prev, c))
            self.pool.append(nn.Conv3d(c, c, kernel_size=2, stride=2))
            prev = c

        bottleneck_ch = min(max_ch, chs[-1] * 2)
        self.bottleneck = ConvBlock(chs[-1], bottleneck_ch)
        self.trans = TransformerBottleneck3D(
            channels=bottleneck_ch,
            depth=trans_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            posenc_type=posenc_type,
        )

        for i in reversed(range(levels)):
            in_c = (bottleneck_ch if i == levels - 1 else chs[i + 1])
            self.up.append(nn.ConvTranspose3d(in_c, chs[i], kernel_size=2, stride=2))
            self.upc.append(ConvBlock(chs[i] * 2, chs[i]))

        self.head = nn.Conv3d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for blk, pool in zip(self.down, self.pool):
            x = blk(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        x = self.trans(x)

        for up, blk, skip in zip(self.up, self.upc, reversed(skips)):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                dz = skip.shape[-3] - x.shape[-3]
                dy = skip.shape[-2] - x.shape[-2]
                dx = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, dx, 0, dy, 0, dz))
            x = torch.cat([x, skip], dim=1)
            x = blk(x)

        return self.head(x)
