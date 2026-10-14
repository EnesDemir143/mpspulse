"""Minimal torchvision-free model definitions for profiling (needs torch).

Hand-rolled CNN/ViT/UNet (~150 lines) to avoid the torchvision dependency
weight; shapes mirror resnet50/vit_b16/unet families at small scale.
"""

from __future__ import annotations

from mpspulse.timing import _require_mps

MODELS = ("resnet50", "vit_b16", "unet_tiny")


def _mod():
    _require_mps()
    import torch
    from torch import nn

    return torch, nn


def make_resnet50(num_classes: int = 1000):
    """ResNet-50-shaped CNN: stem + 4 stages of bottleneck blocks."""
    _torch, nn = _mod()

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            mid = out_ch // self.expansion
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, 3, stride, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.down = None
            if stride != 1 or in_ch != out_ch:
                self.down = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x if self.down is None else self.down(x)
            return self.relu(self.conv(x) + identity)

    layers = [3, 4, 6, 3]  # resnet50 depths
    channels = [256, 512, 1024, 2048]
    net = [
        nn.Conv2d(3, 64, 7, 2, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
    ]
    in_ch = 64
    for depth, out_ch in zip(layers, channels):
        net.append(Bottleneck(in_ch, out_ch, stride=2 if in_ch != 64 else 1))
        for _ in range(depth - 1):
            net.append(Bottleneck(out_ch, out_ch))
        in_ch = out_ch
    net += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], num_classes)]
    return nn.Sequential(*net)


def _transformer_block(dim: int, heads: int):
    _torch, nn = _mod()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            return x + self.mlp(self.norm2(x))

    return Block()


def make_vit_b16(
    num_classes: int = 1000,
    image_size: int = 224,
    patch: int = 16,
    dim: int = 768,
    depth: int = 12,
    heads: int = 12,
):
    """ViT-B/16-shaped transformer: patch embed + class token + blocks."""
    torch, nn = _mod()

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            n = (image_size // patch) ** 2
            self.patch = nn.Conv2d(3, dim, patch, patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
            self.blocks = nn.Sequential(*[_transformer_block(dim, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_classes)

        def forward(self, x):
            t = self.patch(x).flatten(2).transpose(1, 2)
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
            t = self.blocks(t + self.pos)
            return self.head(self.norm(t[:, 0]))

    return ViT()


def make_unet_tiny(in_ch: int = 3, classes: int = 2, base: int = 32):
    """Tiny U-Net: 3 down + bottleneck + 3 up with skip concatenation."""
    torch, nn = _mod()

    def block(c_in: int, c_out: int):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            chs = [base * 2**i for i in range(4)]
            self.downs = nn.ModuleList(
                block(c_in, c_out) for c_in, c_out in zip([in_ch, *chs[:-1]], chs[:-1])
            )
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = block(chs[-2], chs[-1])
            self.ups = nn.ModuleList(nn.ConvTranspose2d(c, c // 2, 2, 2) for c in reversed(chs[1:]))
            self.up_blocks = nn.ModuleList(block(c, c // 2) for c in reversed(chs[1:]))
            self.out = nn.Conv2d(base, classes, 1)

        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips)):
                x = blk(torch.cat([up(x), skip], dim=1))
            return self.out(x)

    return UNet()


def make_model(name: str):
    """Build a builtin model by name; ValueError on unknown name."""
    builders = {"resnet50": make_resnet50, "vit_b16": make_vit_b16, "unet_tiny": make_unet_tiny}
    try:
        return builders[name]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}") from None


def input_shape(name: str, batch_size: int) -> tuple[int, ...]:
    """Representative input shape per model family."""
    shapes = {"resnet50": (3, 224, 224), "vit_b16": (3, 224, 224), "unet_tiny": (3, 128, 128)}
    try:
        return (batch_size, *shapes[name])
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}") from None
