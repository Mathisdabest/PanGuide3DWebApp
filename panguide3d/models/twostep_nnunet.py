import torch
import torch.nn as nn
import torch.nn.functional as F

from .nnunet import ConvBlock
from .nnunet_transbottleneck import TransformerBottleneck3D
try:
    from .nnunet_mamba3d import MambaSSMBlock3d
except Exception:  # mamba variant is not shipped with the web-app package
    MambaSSMBlock3d = None


class TwoStepNNUNet(nn.Module):
    """
    Shared encoder with two decoders:
      - pancreas head
      - tumor head conditioned on pancreas prediction
    """
    def __init__(
        self,
        in_ch=1,
        base=32,
        levels=5,
        max_ch=320,
        conditioning="gating",
        alpha_pan_gating=2.0,
        detach_pancreas_for_tumor=False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base = int(base)
        self.levels = int(levels)
        self.max_ch = int(max_ch)
        self.conditioning = str(conditioning)
        self.alpha_pan_gating = float(alpha_pan_gating)
        self.detach_pancreas_for_tumor = bool(detach_pancreas_for_tumor)

        chs = [min(self.max_ch, self.base * (2 ** i)) for i in range(self.levels)]

        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = self.in_ch
        for c in chs:
            self.down.append(ConvBlock(prev, c))
            self.pool.append(nn.Conv3d(c, c, kernel_size=2, stride=2))
            prev = c

        self.bottleneck_ch = min(self.max_ch, chs[-1] * 2)
        self.bottleneck = ConvBlock(chs[-1], self.bottleneck_ch)

        # Pancreas decoder
        self.up_pan = nn.ModuleList()
        self.upc_pan = nn.ModuleList()
        for i in reversed(range(self.levels)):
            in_c = (self.bottleneck_ch if i == self.levels - 1 else chs[i + 1])
            self.up_pan.append(nn.ConvTranspose3d(in_c, chs[i], kernel_size=2, stride=2))
            self.upc_pan.append(ConvBlock(chs[i] * 2, chs[i]))
        self.head_pan = nn.Conv3d(chs[0], 1, 1)

        # Tumor decoder
        self.up_tum = nn.ModuleList()
        self.upc_tum = nn.ModuleList()
        for i in reversed(range(self.levels)):
            in_c = (self.bottleneck_ch if i == self.levels - 1 else chs[i + 1])
            self.up_tum.append(nn.ConvTranspose3d(in_c, chs[i], kernel_size=2, stride=2))
            self.upc_tum.append(ConvBlock(chs[i] * 2, chs[i]))
        self.head_tum = nn.Conv3d(chs[0], 1, 1)

        if self.conditioning == "roi_concat":
            self.roi_stem = nn.Conv3d(self.in_ch + 1, self.in_ch, kernel_size=1, bias=False)
        else:
            self.roi_stem = None

        if self.conditioning == "xattn":
            self.bias_convs = nn.ModuleList()
            for i in reversed(range(self.levels)):
                self.bias_convs.append(nn.Conv3d(1, chs[i], kernel_size=1, bias=True))
        else:
            self.bias_convs = None

    def _encode(self, x):
        skips = []
        for blk, pool in zip(self.down, self.pool):
            x = blk(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        return x, skips

    def _decode(self, x, skips, up_list, upc_list, head, p_pan=None, bias_convs=None):
        for up, blk, skip, bias_conv in zip(
            up_list,
            upc_list,
            reversed(skips),
            bias_convs if bias_convs is not None else [None] * len(up_list),
        ):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                dz = skip.shape[-3] - x.shape[-3]
                dy = skip.shape[-2] - x.shape[-2]
                dx = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, dx, 0, dy, 0, dz))
            x = torch.cat([x, skip], dim=1)
            x = blk(x)
            if bias_conv is not None and p_pan is not None:
                p_ds = F.interpolate(p_pan, size=x.shape[-3:], mode="trilinear", align_corners=False)
                x = x + bias_conv(p_ds)
        return head(x)

    def _gate(self, feat, p_pan):
        if self.alpha_pan_gating == 0.0:
            return feat
        p_ds = F.interpolate(p_pan, size=feat.shape[-3:], mode="trilinear", align_corners=False)
        return feat * (1.0 + self.alpha_pan_gating * p_ds)

    def forward(self, x):
        # Pancreas path
        bottleneck, skips = self._encode(x)
        logits_pancreas = self._decode(bottleneck, skips, self.up_pan, self.upc_pan, self.head_pan)

        p_pan = torch.sigmoid(logits_pancreas)
        if self.detach_pancreas_for_tumor:
            p_pan = p_pan.detach()

        # Tumor path
        if self.conditioning == "roi_concat":
            x2 = torch.cat([x, p_pan], dim=1)
            x2 = self.roi_stem(x2)
            bottleneck_t, skips_t = self._encode(x2)
        else:
            bottleneck_t, skips_t = bottleneck, skips

        if self.conditioning == "gating":
            bottleneck_t = self._gate(bottleneck_t, p_pan)
            skips_t = [self._gate(s, p_pan) for s in skips_t]

        logits_tumor = self._decode(
            bottleneck_t,
            skips_t,
            self.up_tum,
            self.upc_tum,
            self.head_tum,
            p_pan=p_pan,
            bias_convs=self.bias_convs,
        )

        logits = torch.cat([logits_pancreas, logits_tumor], dim=1)
        return {
            "logits_pancreas": logits_pancreas,
            "logits_tumor": logits_tumor,
            "logits": logits,
        }


class TwoStepNNUNetTransBottleneck(TwoStepNNUNet):
    """
    Two-step NNUNet with transformer bottleneck.
    """
    def __init__(
        self,
        in_ch=1,
        base=32,
        levels=5,
        max_ch=320,
        conditioning="gating",
        alpha_pan_gating=2.0,
        detach_pancreas_for_tumor=False,
        trans_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        posenc_type="sincos3d",
    ):
        super().__init__(
            in_ch=in_ch,
            base=base,
            levels=levels,
            max_ch=max_ch,
            conditioning=conditioning,
            alpha_pan_gating=alpha_pan_gating,
            detach_pancreas_for_tumor=detach_pancreas_for_tumor,
        )
        ch_last = int(self.bottleneck_ch)
        self.trans = TransformerBottleneck3D(
            channels=ch_last,
            depth=trans_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            posenc_type=posenc_type,
        )

    def _encode(self, x):
        skips = []
        for blk, pool in zip(self.down, self.pool):
            x = blk(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        x = self.trans(x)
        return x, skips
