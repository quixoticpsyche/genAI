"""GPU-batched corruption simulator, reverse-engineered from train_corrupt/.

Operates on float tensors in [0,1], shape [B,3,32,32], all ops batched with
per-image random parameters. Order mirrors what the montage shows: geometry
first, then photometric, then additive noise / masks painted on top (noise
appears *over* the black rotation wedge in real samples, so it must come last).

Returns (corrupt, theta) - theta in degrees, 0 for unrotated images, used as
the exact supervision signal for the angle head.
"""
import math, torch, torch.nn.functional as F


def _rand(b, lo, hi, dev):
    return torch.rand(b, device=dev) * (hi - lo) + lo


def _bern(b, p, dev):
    return (torch.rand(b, device=dev) < p)


def rotate(x, theta_deg):
    """Rotate by theta degrees about centre, zero fill. Returns (rotated, validity)."""
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


def gauss_blur(x, sigma):
    """Per-image gaussian blur via a 5-tap separable kernel."""
    b = x.shape[0]
    r = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    k = torch.exp(-(r[None, :] ** 2) / (2 * sigma[:, None] ** 2 + 1e-8))
    k = k / k.sum(1, keepdim=True)                       # [B,5]
    xk = k.view(b, 1, 1, 5).expand(b, 3, 1, 5).reshape(b * 3, 1, 1, 5)
    yk = k.view(b, 1, 5, 1).expand(b, 3, 5, 1).reshape(b * 3, 1, 5, 1)
    v = x.reshape(1, b * 3, *x.shape[2:])
    v = F.conv2d(F.pad(v, (2, 2, 0, 0), mode='reflect'), xk, groups=b * 3)
    v = F.conv2d(F.pad(v, (0, 0, 2, 2), mode='reflect'), yk, groups=b * 3)
    return v.reshape(x.shape)


def rgb2hsv_shift(x, dh, ds):
    """Cheap hue rotation + saturation scale in YIQ-ish space (avoids full HSV)."""
    r, g, bl = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * bl
    i = 0.596 * r - 0.274 * g - 0.322 * bl
    q = 0.211 * r - 0.523 * g + 0.312 * bl
    c, s = torch.cos(dh)[:, None, None], torch.sin(dh)[:, None, None]
    sc = ds[:, None, None]
    i2, q2 = (i * c - q * s) * sc, (i * s + q * c) * sc
    return torch.stack([y + 0.956 * i2 + 0.621 * q2,
                        y - 0.272 * i2 - 0.647 * q2,
                        y - 1.106 * i2 + 1.703 * q2], 1)


def paint_boxes(x, active, sev, n_hi=5, s_lo=2, s_hi=10):
    """Solid-colour rectangular masks (CoarseDropout with random fill)."""
    b, _, H, W = x.shape
    dev = x.device
    out = x
    for j in range(n_hi):
        draw = active & (torch.rand(b, device=dev) < (0.25 + 0.6 * sev))
        span = (s_lo + (s_hi - s_lo) * sev).long().clamp(min=s_lo + 1)
        h = (torch.rand(b, device=dev) * (span - s_lo)).long() + s_lo
        w = (torch.rand(b, device=dev) * (span - s_lo)).long() + s_lo
        y0 = (torch.rand(b, device=dev) * (H - h).clamp(min=1)).long()
        x0 = (torch.rand(b, device=dev) * (W - w).clamp(min=1)).long()
        yy = torch.arange(H, device=dev)[None, :, None]
        xx = torch.arange(W, device=dev)[None, None, :]
        m = ((yy >= y0[:, None, None]) & (yy < (y0 + h)[:, None, None]) &
             (xx >= x0[:, None, None]) & (xx < (x0 + w)[:, None, None]))
        m = m & draw[:, None, None]
        # fill: mostly saturated random colour, sometimes black / grey
        col = torch.rand(b, 3, 1, 1, device=dev)
        kind = torch.rand(b, 1, 1, 1, device=dev)
        col = torch.where(kind < 0.25, torch.zeros_like(col), col)
        col = torch.where((kind >= 0.25) & (kind < 0.4),
                          torch.rand(b, 1, 1, 1, device=dev).expand(-1, 3, -1, -1), col)
        out = torch.where(m[:, None], col, out)
    return out


def impulse(x, active, dens):
    """Coloured salt-and-pepper: random pixels replaced by saturated colours."""
    b, _, H, W = x.shape
    dev = x.device
    hit = (torch.rand(b, 1, H, W, device=dev) < dens[:, None, None, None]) & active[:, None, None, None]
    col = torch.rand(b, 3, H, W, device=dev)
    col = torch.where(torch.rand(b, 3, 1, 1, device=dev) < 0.35, (col > 0.5).float(), col)
    return torch.where(hit, col, x)


def streaks(x, active, n=3):
    """Horizontal row corruption: a few rows shifted or replaced."""
    b, _, H, W = x.shape
    dev = x.device
    out = x
    for _ in range(n):
        draw = active & _bern(b, 0.5, dev)
        y0 = torch.randint(0, H - 1, (b,), device=dev)
        th = torch.randint(1, 3, (b,), device=dev)
        yy = torch.arange(H, device=dev)[None, :]
        m = ((yy >= y0[:, None]) & (yy < (y0 + th)[:, None])) & draw[:, None]
        shift = int(torch.randint(2, 7, (1,)).item())
        rolled = torch.roll(out, shift, dims=-1)
        out = torch.where(m[:, None, :, None], rolled, out)
    return out


DEFAULT_P = dict(rot=0.235, boxes=0.11, impulse=0.12, gnoise=0.45, blur=0.30,
                 down=0.12, bc=0.42, hue=0.18, streak=0.04, clean=0.03)

# Real per-image identity MSE spans ~4 orders of magnitude with a smooth ramp
# (10th pct 3.6, median 148, 90th pct 1379). Fixed magnitudes cannot produce
# that shape, so every op's strength is scaled by a per-image log-uniform
# severity s, which makes MSE roughly log-uniform too.
SEV_LO, SEV_HI = 0.085, 0.72


def corrupt(clean, p=None, sev=None):
    """clean: [B,3,32,32] float in [0,1]. -> (corrupt, theta_deg)"""
    P = dict(DEFAULT_P); P.update(p or {})
    x = clean
    b, dev = x.shape[0], x.device
    keep = _bern(b, P['clean'], dev)
    lo, hi = (sev or (SEV_LO, SEV_HI))
    s = torch.exp(_rand(b, math.log(lo), math.log(hi), dev))     # log-uniform severity
    s = s * (~keep).float()

    # ---- photometric (before geometry: real wedges are *exactly* 0, so no
    #      brightness bias may be added after the rotation) ----
    m = (_bern(b, P['bc'], dev) & ~keep).float()
    gain = 1 + m * s * _rand(b, -0.45, 0.25, dev)
    bias = m * s * _rand(b, -0.08, 0.28, dev)
    gam = 1 + m * s * _rand(b, -0.4, 0.55, dev)
    x = (x.clamp(0, 1) ** gam[:, None, None, None]) * gain[:, None, None, None] \
        + bias[:, None, None, None]
    x = x.clamp(0, 1)

    m = _bern(b, P['hue'], dev) & ~keep
    x = torch.where(m[:, None, None, None],
                    rgb2hsv_shift(x, s * _rand(b, -0.7, 0.7, dev),
                                  1 + s * _rand(b, -0.7, 0.7, dev)), x).clamp(0, 1)

    # ---- geometry ----
    do_rot = _bern(b, P['rot'], dev) & ~keep
    # rotation magnitude is independent of the other degradations in the real data:
    # observed angles cluster at 3-13 deg regardless of how noisy the image is
    sign = torch.where(_bern(b, 0.5, dev), 1.0, -1.0)
    theta = sign * _rand(b, 3.0, 14.5, dev) * do_rot.float()
    x, valid = rotate(x, theta)
    valid = (valid > 0.999).float()

    # ---- resolution loss ----
    m = _bern(b, P['blur'], dev) & ~keep
    xb = gauss_blur(x, 0.35 + 1.05 * s)
    x = torch.where(m[:, None, None, None], xb, x)

    m = _bern(b, P['down'], dev) & ~keep & (s > 0.45)
    xd = F.interpolate(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False),
                       size=(32, 32), mode='bilinear', align_corners=False)
    x = torch.where(m[:, None, None, None], xd, x)

    # ---- painted-on degradations (after geometry: real noise sits over the wedge) ----
    x = paint_boxes(x, _bern(b, P['boxes'], dev) & ~keep, s)
    x = streaks(x, _bern(b, P['streak'], dev) & ~keep)

    m = (_bern(b, P['gnoise'], dev) & ~keep).float()
    x = x + torch.randn_like(x) * (m * s * 0.11)[:, None, None, None] * valid

    x = x + torch.randn_like(x) * ((~keep).float() * (0.004 + 0.010 * s))[:, None, None, None] * valid
    x = impulse(x, _bern(b, P['impulse'], dev) & ~keep, s * 0.055)

    x = x.clamp(0, 1)
    x = torch.where(keep[:, None, None, None], clean, x)
    theta = torch.where(keep, torch.zeros_like(theta), theta)
    x = (x * 255).round() / 255.0          # real data is stored as PNG
    return x, theta
