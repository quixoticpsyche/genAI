"""Angle-head + UNet restoration model.

Rotation accounts for ~53% of total error, so geometry is handled explicitly:
a small head regresses the rotation angle (supervised exactly on synthetic
samples), the image is de-rotated with a differentiable warp, and the UNet
receives BOTH the de-rotated and the original image. That second path is the
safety valve - when the angle is wrong, or the image was never rotated, the
network still has an unwarped view and degrades gracefully.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def warp(x, theta_deg):
    b = x.shape[0]
    t = theta_deg * math.pi / 180.0
    cos, sin = torch.cos(t), torch.sin(t)
    mat = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    mat[:, 0, 0], mat[:, 0, 1] = cos, -sin
    mat[:, 1, 0], mat[:, 1, 1] = sin, cos
    grid = F.affine_grid(mat, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    ones = torch.ones(b, 1, *x.shape[2:], device=x.device, dtype=x.dtype)
    val = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return out, val


class AngleHead(nn.Module):
    """Predicts rotation in degrees. Gets exact labels from the simulator."""
    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GroupNorm(8, c), nn.SiLU(),        # 16
            nn.Conv2d(c, c * 2, 3, 1, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 3, 2, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),  # 8
            nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),
        )
        self.fc = nn.Linear(c * 4, 1)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h = self.net(x)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.fc(h).squeeze(1) * 15.0        # scaled to the plausible range


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, c); self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1); self.h = heads
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                           v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(b, c, H, W))


class UNet(nn.Module):
    def __init__(self, cin=8, w=(128, 256, 512), nb=2):
        super().__init__()
        w1, w2, w3 = w
        self.stem = nn.Conv2d(cin, w1, 3, 1, 1)
        self.e1 = nn.ModuleList([Block(w1, w1) for _ in range(nb)])
        self.d1 = nn.Conv2d(w1, w2, 3, 2, 1)
        self.e2 = nn.ModuleList([Block(w2, w2) for _ in range(nb)])
        self.a2 = Attn(w2)
        self.d2 = nn.Conv2d(w2, w3, 3, 2, 1)
        self.mid = nn.ModuleList([Block(w3, w3), Attn(w3), Block(w3, w3)])
        self.u2 = nn.Conv2d(w3, w2, 3, 1, 1)
        self.f2 = nn.ModuleList([Block(w2 * 2, w2)] + [Block(w2, w2) for _ in range(nb - 1)])
        self.b2 = Attn(w2)
        self.u1 = nn.Conv2d(w2, w1, 3, 1, 1)
        self.f1 = nn.ModuleList([Block(w1 * 2, w1)] + [Block(w1, w1) for _ in range(nb - 1)])
        self.out_n = nn.GroupNorm(8, w1); self.out = nn.Conv2d(w1, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at identity

    def forward(self, x):
        h1 = self.stem(x)
        for b in self.e1: h1 = b(h1)
        h2 = self.d1(h1)
        for b in self.e2: h2 = b(h2)
        h2 = self.a2(h2)
        h3 = self.d2(h2)
        for m in self.mid: h3 = m(h3)
        g2 = self.u2(F.interpolate(h3, scale_factor=2, mode='nearest'))
        g2 = torch.cat([g2, h2], 1)
        for b in self.f2: g2 = b(g2)
        g2 = self.b2(g2)
        g1 = self.u1(F.interpolate(g2, scale_factor=2, mode='nearest'))
        g1 = torch.cat([g1, h1], 1)
        for b in self.f1: g1 = b(g1)
        return self.out(F.silu(self.out_n(g1)))


class Restorer(nn.Module):
    def __init__(self, w=(128, 256, 512), nb=2):
        super().__init__()
        self.angle = AngleHead()
        self.unet = UNet(cin=8, w=w, nb=nb)

    def forward(self, x):
        """x: corrupt in [0,1], [B,3,32,32]. -> (pred_clean, pred_theta)"""
        blk = (x.sum(1, keepdim=True) < 0.012).float()      # the rotation wedge
        th = self.angle(torch.cat([x, blk], 1))
        # warp with a detached angle: the head learns from its own exact
        # supervision, not from noisy grid_sample gradients
        wx, val = warp(x, -th.detach())
        inp = torch.cat([x, blk, wx, val], 1)
        return (x + self.unet(inp)).clamp(0, 1), th
